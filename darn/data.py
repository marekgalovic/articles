import tensorflow as tf

from darn import TARGET_STOCK
from darn.utils import input_reader, z_score, inverse_z_score, pct_change, normalize_to_unit_sum


N_PAST = 60
N_FUTURE = 5

TARGET = 'target_%s_close' % TARGET_STOCK
PREVIOUS_TARGET = '%s_close' % TARGET_STOCK
STOCKS = ['aapl', 'amzn', 'googl', 'fb', 'msft', 'ibm', 'shop']
FEATURES = ['open', 'close', 'low', 'high', 'volume', 'spread']
TIMESERIES_FEATURES = ['%s_%s' % (stock, feature) for feature in FEATURES for stock in STOCKS]
STATIC_FEATURES = ['%s_mean_%s' % (stock, feature) for feature in FEATURES for stock in STOCKS]
TIME_EMBEDDINGS = ['month', 'week_of_year', 'day_of_week', 'idx']


def extend_parsed(parsed):
    for stock in STOCKS:
        parsed['%s_spread' % stock] = tf.abs(parsed['%s_high' % stock] - parsed['%s_low' % stock])
        # Static
        for feature in FEATURES:
            parsed['%s_mean_%s' % (stock, feature)] = tf.reduce_mean(parsed['%s_%s' % (stock, feature)], -1)

    return parsed


def parse_fn(no_target=False):
    def _callable(example):
        features = {
            # Timeseries Features
            'month': tf.FixedLenFeature((N_PAST), tf.int64),
            'week_of_year': tf.FixedLenFeature((N_PAST), tf.int64),
            'day_of_week': tf.FixedLenFeature((N_PAST), tf.int64),
            # Target timestamps
            'target_month': tf.FixedLenFeature((N_FUTURE), tf.int64),
            'target_week_of_year': tf.FixedLenFeature((N_FUTURE), tf.int64),
            'target_day_of_week': tf.FixedLenFeature((N_FUTURE), tf.int64),
        }

        for stock in STOCKS:
            features['%s_open' % stock] = tf.FixedLenFeature((N_PAST), tf.float32)
            features['%s_close' % stock] = tf.FixedLenFeature((N_PAST), tf.float32)
            features['%s_low' % stock] = tf.FixedLenFeature((N_PAST), tf.float32)
            features['%s_high' % stock] = tf.FixedLenFeature((N_PAST), tf.float32)
            features['%s_volume' % stock] = tf.FixedLenFeature((N_PAST), tf.int64)

        if not no_target:
            features[TARGET] = tf.FixedLenFeature((N_FUTURE), tf.float32)
        
        parsed = tf.parse_single_example(example, features)
        parsed = extend_parsed(parsed)

        # Value indices
        parsed['idx'] = tf.range(N_PAST, dtype=tf.int64)
        parsed['target_idx'] = tf.range(N_FUTURE, dtype=tf.int64)

        return parsed

    return _callable


def cast_and_normalize(value, stats):
    value = tf.cast(value, tf.float32)

    return z_score(tf.log1p(value), stats['mean_log'], stats['stddev_log'])


def rescale_predictions(stats):
    def _callable(features, value):
        return tf.expand_dims(features['last_known_target_value'], -1) * value

    return _callable


def prepare_features_fn(feature_stats):
    def _callable(samples):
        timeseries = tf.stack([
            cast_and_normalize(samples[col], feature_stats[col])
            for col in TIMESERIES_FEATURES
        ] + [
            tf.clip_by_value(pct_change(samples[col]), -10, 10) / 10.0
            for col in TIMESERIES_FEATURES
        ] + [
            tf.clip_by_value(normalize_to_unit_sum(samples[col]), 0, 1)
            for col in TIMESERIES_FEATURES
        ], -1)

        static = tf.stack([
            cast_and_normalize(samples[col], feature_stats[col])
            for col in STATIC_FEATURES
        ], -1)

        features = {
            'timeseries': timeseries,
            'static': static,
            'input_length': tf.reduce_sum(tf.cast(tf.greater(samples['month'], 0), tf.int32), -1, name='input_length'),
            'initial_decoder_value': samples[PREVIOUS_TARGET][:, -1] / samples[PREVIOUS_TARGET][:, -N_FUTURE],
            'last_known_target_value': samples[PREVIOUS_TARGET][:, -1]
        }

        for embedding in TIME_EMBEDDINGS:
            features[embedding] = samples[embedding]
            features['target_%s' % embedding] = samples['target_%s' % embedding]

        if TARGET in samples:
            return features, {
                'raw': samples[TARGET],
                'rescaled': samples[TARGET] / tf.expand_dims(features['last_known_target_value'], -1)
            }

        return features

    return _callable


def train_input_fn(path, feature_stats, epochs=1, batch_size=128):
    def _callable():
        return input_reader(path, parse_fn(), epochs=epochs, batch_size=batch_size)\
            .map(prepare_features_fn(feature_stats))\
            .make_one_shot_iterator()\
            .get_next()

    return _callable


def eval_input_fn(path, feature_stats, batch_size=128):
    def _callable():
        return input_reader(path, parse_fn(), batch_size=batch_size)\
            .map(prepare_features_fn(feature_stats))\
            .make_one_shot_iterator()\
            .get_next()

    return _callable


def serving_input_receiver_fn(feature_stats, batch_size=None):
    def _callable():
        receiver_tensors = {
            # Timeseries features
            'month': tf.placeholder(tf.int64, [batch_size, N_PAST], name='ph/month'),
            'week_of_year': tf.placeholder(tf.int64, [batch_size, N_PAST], name='ph/week_of_year'),
            'day_of_week': tf.placeholder(tf.int64, [batch_size, N_PAST], name='ph/day_of_week'),
            # Target timestamps
            'target_month': tf.placeholder(tf.int64, [batch_size, N_FUTURE], name='ph/target_month'),
            'target_week_of_year': tf.placeholder(tf.int64, [batch_size, N_FUTURE], name='ph/target_week_of_year'),
            'target_day_of_week': tf.placeholder(tf.int64, [batch_size, N_FUTURE], name='ph/target_day_of_week'),
        }

        for stock in STOCKS:
            receiver_tensors['%s_open' % stock] = tf.placeholder(tf.float32, [batch_size, N_PAST], name='ph/%s_open' % stock)
            receiver_tensors['%s_close' % stock] = tf.placeholder(tf.float32, [batch_size, N_PAST], name='ph/%s_close' % stock)
            receiver_tensors['%s_low' % stock] = tf.placeholder(tf.float32, [batch_size, N_PAST], name='ph/%s_low' % stock)
            receiver_tensors['%s_high' % stock] = tf.placeholder(tf.float32, [batch_size, N_PAST], name='ph/%s_high' % stock)
            receiver_tensors['%s_volume' % stock] = tf.placeholder(tf.int64, [batch_size, N_PAST], name='ph/%s_volume' % stock)

        receiver_tensors = extend_parsed(receiver_tensors)

        # Value indices
        tile_spec = [tf.shape(receiver_tensors['month'])[0], 1]
        receiver_tensors['idx'] = tf.tile(tf.expand_dims(tf.range(N_PAST, dtype=tf.int64), 0), tile_spec)
        receiver_tensors['target_idx'] = tf.tile(tf.expand_dims(tf.range(N_FUTURE, dtype=tf.int64), 0), tile_spec)

        features = prepare_features_fn(feature_stats)(receiver_tensors)

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    return _callable
