import tensorflow as tf

from darn.modules import embeddings, convolve_timeseries, project_timeseries, inputs_encoder, context, decoder


class DARN(tf.estimator.Estimator):
    '''Deep Attentional AutoRegressive Network'''

    def __init__(
        self,
        rescale_predictions,
        embeddings=[],
        input_timestamps=[],
        target_timestamps=[],
        timeseries_embeddings=[],
        static_embeddings=[],
        target_length=None,
        convolution_kernel_size=None,
        target_grouping=None,
        **kwargs):
        assert (target_length is not None) and isinstance(target_length, int)

        self._rescale_predictions = rescale_predictions
        self._embeddings = embeddings
        self._input_timestamps = input_timestamps
        self._target_timestamps = target_timestamps
        self._timeseries_embeddings = timeseries_embeddings
        self._static_embeddings = static_embeddings
        self._target_length = target_length
        self._convolution_kernel_size = convolution_kernel_size
        self._target_grouping = target_grouping

        super(DARN, self).__init__(
            model_fn=self._model_fn,
            **kwargs,
        )

    def _model_fn(self, features, labels, mode, params):
        hidden_size = int(params.get('hidden_size', 256))
        embedding_size = int(params.get('embedding_size', 16))
        encoder_residual_layers = int(params.get('encoder_residual_layers', 0))
        decoder_layers = int(params.get('decoder_layers', 0))
        previous_output_sampling_p = float(params.get('previous_output_sampling_p', 0))
        dropout = float(params.get('dropout', 0))
        reg_scale = float(params.get('reg_scale', 0))
        # Optimizer
        learning_rate = float(params.get('learning_rate', 1e-3))
        learning_rate_decay = float(params.get('learning_rate_decay', 0.97))
        learning_rate_decay_steps = int(params.get('learning_rate_decay_steps', 7500))

        inputs = features['timeseries']
        input_length = features['input_length']
        (input_timestamps, target_timestamps, timeseries_embeddings, static_embeddings) = self._embed(features, embedding_size)
        input_embeddings = input_timestamps if (timeseries_embeddings is None) else tf.concat([input_timestamps, timeseries_embeddings], -1)

        if self._convolution_kernel_size is not None:
            (inputs, input_embeddings, input_length) = convolve_timeseries(
                inputs,
                input_embeddings,
                input_length,
                filters=hidden_size,
                kernel_size=self._convolution_kernel_size,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale)
            )
        else:
            inputs = project_timeseries(
                inputs,
                size=hidden_size,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale)
            )

        (encoder_outputs, encoder_state) = inputs_encoder(
            inputs,
            input_embeddings,
            input_length,
            size=hidden_size,
            num_residual_layers=encoder_residual_layers,
            dropout=dropout
        )

        ctx = context(
            features.get('static'),
            encoder_state,
            static_embeddings,
            size=hidden_size,
            dropout=dropout,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale)
        )

        raw_predictions = decoder(
            ctx,
            features['initial_decoder_value'],
            target_timestamps,
            encoder_outputs,
            input_length,
            size=hidden_size,
            target_length=self._target_length,
            num_layers=decoder_layers,
            target_values=(labels['rescaled'] if (labels is not None) else None),
            previous_output_sampling_p=previous_output_sampling_p,
            dropout=dropout,
            timestamps_grouping_factor=self._target_grouping,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_scale),
        )

        rescaled_predictions = self._rescale_predictions(features, raw_predictions)

        if (mode == tf.estimator.ModeKeys.TRAIN) or (mode == tf.estimator.ModeKeys.EVAL):
            # squared_error = tf.reduce_mean(tf.square(labels['rescaled'] - raw_predictions), 0)
            # weight = 1 + tf.range(self._target_length, dtype=tf.float32) / 10.0
            # loss = tf.reduce_sum(squared_error * weight)

            loss = tf.losses.mean_squared_error(
                labels=labels['rescaled'],
                predictions=raw_predictions
            )

            lr = tf.train.exponential_decay(
                learning_rate,
                global_step=tf.train.get_global_step(),
                decay_steps=learning_rate_decay_steps,
                decay_rate=learning_rate_decay
            )
            tf.summary.scalar('learning_rate', lr)

            train_op = tf.train.AdamOptimizer(lr).minimize(loss, global_step=tf.train.get_global_step())

            daily_mae = tf.reduce_mean(tf.abs(labels['raw'] - rescaled_predictions), 0)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops={
                'mae': tf.metrics.mean_absolute_error(labels['raw'], rescaled_predictions),
                'daily_mae': tf.metrics.mean(daily_mae)
            })

        return tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'raw_predictions': raw_predictions,
                'rescaled_predictions': rescaled_predictions
            },
            export_outputs={
                'predictions': tf.estimator.export.PredictOutput(rescaled_predictions)
            }
        )

    def _embed(self, features, embedding_size):
        embedded = embeddings(features, self._embeddings, embedding_size=embedding_size)

        input_timestamps = tf.concat([
            embedded[feature]
            for feature in self._input_timestamps
        ], -1, name='input_timestamps')

        target_timestamps = tf.concat([
            embedded[feature]
            for feature in self._target_timestamps
        ], -1, name='target_timestamps')

        timeseries_embeddings = None
        if len(self._timeseries_embeddings) > 0:
            timeseries_embeddings = tf.concat([
                embedded[feature]
                for feature in self._timeseries_embeddings
            ], -1, name='timeseries_embeddings')

        static_embeddings = None
        if len(self._static_embeddings) > 0:
            static_embeddings = tf.concat([
                embedded[feature]
                for feature in self._static_embeddings
            ], -1, name='static_embeddings')

        return input_timestamps, target_timestamps, timeseries_embeddings, static_embeddings
