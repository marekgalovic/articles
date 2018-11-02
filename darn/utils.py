import json

import numpy as np
import tensorflow as tf


def load_json(fullpath):
    f = tf.gfile.Open(fullpath, 'r')
    return json.load(f)


def write_json(fullpath, data):
    f = tf.gfile.Open(fullpath, 'wb')
    json.dump(data, f)
    f.close()


def clip(x, min_value, max_value):
    '''
    Numpy style clip
    '''
    if (min_value is None) and (max_value is None):
        raise ValueError("No thresholds provided")

    if min_value is None:
        min_value = tf.reduce_min(x)

    if max_value is None:
        max_value = tf.reduce_max(x)

    return tf.clip_by_value(x, min_value, max_value)


def pct_change(values, impute_nan=True):
    if len(values.get_shape()) != 2:
        raise ValueError('Values have to be a rank 2 tensor')

    values = tf.cast(values, tf.float32)
    pad = tf.zeros([tf.shape(values)[0], 1])
    shifted_values = tf.concat([pad, values[:, :-1]], -1)

    change = (values - shifted_values) / shifted_values

    if impute_nan:
        change = tf.where(
            tf.logical_or(tf.is_inf(change), tf.is_nan(change)),
            tf.zeros_like(change),
            change
        )

    return change


def normalize_to_unit_sum(values):
    if len(values.get_shape()) != 2:
        raise ValueError('Values have to be a rank 2 tensor')

    values = tf.cast(clip(values, 0, None), tf.float32)
    cumulative = tf.reduce_sum(values, -1)

    return tf.where(
        tf.equal(cumulative, 0),
        tf.ones_like(values) / tf.cast(tf.shape(values)[-1], tf.float32),
        values / tf.expand_dims(cumulative, -1)
    )


def z_score(x, mean, stddev):
    assert stddev != 0

    return (x - mean) / stddev


def inverse_z_score(x, mean, stddev):
    assert stddev != 0

    return x * stddev + mean


def input_reader(path, parse_fn, epochs=1, batch_size=None, shuffle=False, prefetch=1):
    data = tf.data.Dataset\
        .list_files(path)\
        .interleave(lambda f: tf.data.TFRecordDataset(f).map(parse_fn, num_parallel_calls=1), cycle_length=8, block_length=32)

    if shuffle:
        data = data.shuffle(1000)

    data = data.repeat(epochs)

    if batch_size is not None:
        data = data.batch(batch_size)

    return data.prefetch(prefetch)


def compute_feature_stats(dataset, fields=[]):
    assert isinstance(dataset, tf.data.Dataset)

    it = dataset\
        .make_one_shot_iterator()\
        .get_next()

    feature_stats = {
        feature: {'count': 0, 'sum': 0.0, 'sum_sq': 0.0, 'sum_log': 0.0, 'sum_sq_log': 0.0}
        for feature in fields
    }

    with tf.Session() as sess:
        while True:
            try:
                features = sess.run(it)

                for (feature, value) in features.items():
                    if feature not in feature_stats:
                        continue

                    if len(value.shape) > 0:
                        for val in value:
                            feature_stats[feature]['count'] += 1
                            feature_stats[feature]['sum'] += val
                            feature_stats[feature]['sum_sq'] += (val ** 2)
                            feature_stats[feature]['sum_log'] += np.log1p(val)
                            feature_stats[feature]['sum_sq_log'] += (np.log1p(val) ** 2)
                    else:
                        feature_stats[feature]['count'] += 1
                        feature_stats[feature]['sum'] += value
                        feature_stats[feature]['sum_sq'] += (value ** 2)
                        feature_stats[feature]['sum_log'] += np.log1p(value)
                        feature_stats[feature]['sum_sq_log'] += (np.log1p(value) ** 2)

            except tf.errors.OutOfRangeError:
                break

    for (feature, raw) in feature_stats.items():
        feature_stats[feature] = {
            'mean': float(raw['sum']) / raw['count'],
            'stddev': np.sqrt((float(raw['sum_sq']) / raw['count']) - ((float(raw['sum']) / raw['count']) ** 2)) ,
            'mean_log': float(raw['sum_log']) / raw['count'],
            'stddev_log': np.sqrt((float(raw['sum_sq_log']) / raw['count']) - ((float(raw['sum_log']) / raw['count']) ** 2))
        }
        
    return feature_stats
