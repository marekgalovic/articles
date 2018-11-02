from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, lookup

from darn.utils import clip


class Embedding(namedtuple('Embedding', [
    'name', 'fields', 'size', 'table'
])):

    def __new__(cls, name, fields, size=None, table=None):
        assert isinstance(name, str)
        assert isinstance(fields, list)
        assert (size is not None) or (table is not None)

        return super(Embedding, cls).__new__(cls, name, fields, size, table)

    @property
    def length(self):
        return self.size or len(self.table)


def embeddings(features, configs, embedding_size=None):
    assert (embedding_size is not None) and isinstance(embedding_size, int)
    
    embedded = {}
    with tf.variable_scope('embeddings'):
        for config in configs:
            embeddings = tf.get_variable(
                name=config.name,
                shape=[config.length, embedding_size],
                dtype=tf.float32,
                trainable=True,
                initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5, dtype=tf.float32),
            )

            if config.table is not None:
                lookup_table = lookup.index_table_from_tensor(
                    mapping=tf.constant(config.table, dtype=tf.string),
                    default_value=0
                )

                for field in config.fields:
                    embedded[field] = tf.nn.embedding_lookup(embeddings, lookup_table.lookup(features[field]))

            else:
                for field in config.fields:
                    embedded[field] = tf.nn.embedding_lookup(embeddings, features[field])

    return embedded


def convolve_timeseries(inputs, embeddings, sequence_length, kernel_size=None, filters=None, activation=tf.nn.tanh, kernel_regularizer=None):
    assert (kernel_size is not None) and isinstance(kernel_size, int)
    assert (filters is not None) and isinstance(filters, int)

    with tf.variable_scope('convolve_timeseries'):
        sequence_length = tf.ceil(tf.cast(sequence_length, tf.float32) / float(kernel_size))
        sequence_length = tf.cast(clip(sequence_length, 1, None), tf.int32)

        def _pad_sequence(sequence):
            batch_size, max_t, v_dim = tf.unstack(tf.shape(sequence))
            pad_size = kernel_size - (max_t % kernel_size)
            pad_size = tf.where(tf.greater_equal(pad_size, kernel_size), tf.zeros_like(pad_size), pad_size)

            padding = tf.zeros([batch_size, pad_size, v_dim], dtype=sequence.dtype)

            return tf.concat([sequence, padding], 1)

        inputs = _pad_sequence(inputs)
        embeddings = _pad_sequence(embeddings)

        convolved_inputs = tf.layers.conv1d(
            inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            name='inputs'
        )

        grouped_embeddings_shape = [-1, tf.shape(convolved_inputs)[1], kernel_size * embeddings.shape.as_list()[-1]]
        grouped_embeddings = tf.reshape(embeddings, grouped_embeddings_shape, name='embeddings')

        return convolved_inputs, grouped_embeddings, sequence_length


def project_timeseries(inputs, size=None, activation=tf.nn.tanh, kernel_regularizer=None):
    assert (size is not None) and isinstance(size, int)

    with tf.variable_scope('project_timeseries'):
        return tf.layers.dense(
            inputs,
            size,
            kernel_regularizer=kernel_regularizer,
            activation=activation,
            name='inputs'
        )


def inputs_encoder(inputs, embeddings, sequence_length, size=None, activation=tf.nn.tanh, num_residual_layers=0, dropout=0.0):
    assert (size is not None) and isinstance(size, int)
    assert isinstance(num_residual_layers, int) and (num_residual_layers >= 0)

    with tf.variable_scope('inputs_encoder'):
        encoder_inputs = tf.concat([inputs, embeddings], -1)

        stacked_cell = rnn.MultiRNNCell([
            rnn.DropoutWrapper(rnn.GRUCell(size, activation=activation), output_keep_prob=(1. - dropout))
        ] + [
            rnn.DropoutWrapper(rnn.ResidualWrapper(rnn.GRUCell(size, activation=activation)), output_keep_prob=(1. - dropout))
            for _ in range(num_residual_layers)
        ])

        _outputs, _state = tf.nn.dynamic_rnn(
            cell=stacked_cell,
            inputs=encoder_inputs,
            sequence_length=sequence_length,
            dtype=tf.float32
        )

        return _outputs, tf.concat(_state, -1)


def context(static_features, inputs_encoder_state, embeddings, size=None, activation=tf.nn.tanh, dropout=0.0, kernel_regularizer=None):
    assert (size is not None) and isinstance(size, int)

    with tf.variable_scope('context'):
        if static_features is not None:
            static_features = tf.layers.dense(
                static_features if (embeddings is None) else tf.concat([static_features, embeddings], -1),
                size,
                kernel_regularizer=kernel_regularizer,
                activation=activation
            )
        else:
            static_features = embeddings

        context = tf.layers.dense(
            (tf.concat([static_features, inputs_encoder_state], -1) if (static_features is not None) else inputs_encoder_state),
            size,
            kernel_regularizer=kernel_regularizer,
            activation=activation
        )

        return tf.layers.dropout(context, rate=dropout, name='context')


def linear_regression(kernel_regularizer=None):
    layer = tf.layers.Dense(1, kernel_regularizer=kernel_regularizer)

    def _callable(x):
        return tf.squeeze(layer(x), -1)

    return _callable


def attention(memory, memory_length=None, size=None, activation=tf.nn.tanh):
    assert (size is not None) and isinstance(size, int)

    alignments_layer = tf.layers.Dense(1)
    next_state_layer = tf.layers.Dense(size, activation=activation)

    def _callable(state):
        tiled_state = tf.tile(
            tf.expand_dims(state, 1),
            [1, tf.shape(memory)[1], 1]
        )
        alignment_inputs = tf.concat([tiled_state, memory], -1)

        alignments = tf.squeeze(alignments_layer(alignment_inputs), -1)
        if memory_length is not None:
            alignments = tf.where(tf.sequence_mask(memory_length), alignments, tf.ones_like(alignments) * -np.inf)

        weights = tf.expand_dims(tf.nn.softmax(alignments), -1)
        context = tf.reduce_sum(weights * memory, 1)

        return next_state_layer(tf.concat([state, context], -1))

    return _callable


def decoder(initial_state, initial_value, timestamps, attention_memory, attention_memory_length, target_length=None, size=None, activation=tf.nn.tanh, num_layers=1, dropout=0.0, previous_output_sampling_p=1.0, target_values=None, kernel_regularizer=None, timestamps_grouping_factor=None):
    assert (size is not None) and isinstance(size, int)
    assert (target_length is not None) and isinstance(target_length, int) and (target_length > 0)
    assert (num_layers is not None) and isinstance(num_layers, int) and (num_layers > 0)

    with tf.variable_scope('decoder'):
        batch_size = tf.shape(initial_state)[0]
        sequence_length = tf.ones([batch_size], tf.int32) * target_length

        if timestamps_grouping_factor is not None:
            timestamps = tf.reshape(timestamps, [batch_size, target_length, timestamps_grouping_factor * timestamps.shape.as_list()[-1]])

        timestamps_ta = tf.TensorArray(dtype=tf.float32, size=target_length, name='timestamps_ta')\
            .unstack(tf.transpose(timestamps, [1, 0, 2]))

        if target_values is not None:
            target_values = tf.concat([
                tf.expand_dims(initial_value, -1),
                target_values
            ], -1)
            target_values_ta = tf.TensorArray(dtype=tf.float32, size=target_length + 1, name='target_values_ta')\
                .unstack(tf.transpose(target_values))

        regression_layer = linear_regression(kernel_regularizer=kernel_regularizer)
        attention_layer = attention(attention_memory, attention_memory_length, size=size)

        cell = rnn.MultiRNNCell([
            rnn.DropoutWrapper(rnn.GRUCell(size, activation=activation), output_keep_prob=(1. - dropout))
            for _ in range(num_layers)
        ])

        def _rnn_loop(time, cell_output, cell_state, loop_state):
            finished = (time >= sequence_length)

            if cell_output is None:
                previous_output = initial_value
                next_state = tuple([
                    initial_state
                    for _ in range(num_layers)
                ])
            else:
                previous_output = regression_layer(cell_output)
                next_state = cell_state

            next_state = tuple([
                attention_layer(next_state[0])
            ] + [
                next_state[i + 1]
                for i in range(num_layers - 1)
            ])

            next_timestamp = tf.cond(
                tf.reduce_all(finished),
                lambda: tf.zeros([batch_size, timestamps.shape.as_list()[-1]]),
                lambda: timestamps_ta.read(time)
            )

            if target_values is not None:
                previous_output = tf.where(
                    tf.less_equal(tf.random_uniform([batch_size]), previous_output_sampling_p),
                    previous_output,
                    target_values_ta.read(time),
                    name='target_values_sampling'
                )

            next_input = tf.concat([
                next_timestamp,
                tf.expand_dims(previous_output, -1)
            ], -1)

            return finished, next_input, next_state, cell_output, None

        outputs, _, _ = tf.nn.raw_rnn(cell, _rnn_loop)
        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        
        return tf.reshape(
            regression_layer(outputs),
            [-1, target_length],
            name='outputs'
        )
