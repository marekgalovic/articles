import os
from argparse import ArgumentParser
from datetime import datetime

import tensorflow as tf

from darn.model import DARN
from darn.modules import Embedding
from darn.data import N_PAST, N_FUTURE, TARGET, TIMESERIES_FEATURES, STATIC_FEATURES, TIME_EMBEDDINGS, parse_fn, train_input_fn, eval_input_fn, rescale_predictions
from darn.utils import input_reader, compute_feature_stats, load_json, write_json


def main(args):
    feature_stats = {}
    if not tf.gfile.Exists(args.feature_stats):
        feature_stats = compute_feature_stats(
            input_reader(args.train_data, parse_fn()),
            fields=TIMESERIES_FEATURES + STATIC_FEATURES + [TARGET]
        )

        write_json(args.feature_stats, feature_stats)
    else:
        feature_stats = load_json(args.feature_stats)

    model = DARN(
        rescale_predictions(feature_stats),
        embeddings = [
            Embedding('month_embeddings', ['month', 'target_month'], size=13),
            Embedding('week_of_year_embeddings', ['week_of_year', 'target_week_of_year'], size=54),
            Embedding('day_of_week_embeddings', ['day_of_week', 'target_day_of_week'], size=8),
            Embedding('input_idx_embeddings', ['idx'], size=N_PAST),
            Embedding('target_idx_embeddings', ['target_idx'], size=N_FUTURE),
        ],
        input_timestamps=TIME_EMBEDDINGS,
        target_timestamps=['target_%s' % name for name in TIME_EMBEDDINGS],
        target_length=N_FUTURE,
        convolution_kernel_size=5,
        params = {
            'hidden_size': args.hidden_size,
            'embedding_size': args.embedding_size,
            'encoder_residual_layers': args.encoder_residual_layers,
            'decoder_layers': args.decoder_layers,
            'previous_output_sampling_p': args.previous_output_sampling_p,
            'dropout': args.dropout,
            'reg_scale': args.reg_scale,
        },
        config = tf.estimator.RunConfig(
            model_dir=os.path.join(args.job_dir, datetime.now().strftime('%Y%m%d%H%M%s')),
            save_summary_steps=10,
            keep_checkpoint_max=None,
            save_checkpoints_secs=30,
        )
    )

    tf.estimator.train_and_evaluate(
        model,
        train_spec=tf.estimator.TrainSpec(
            input_fn=train_input_fn(args.train_data, feature_stats, epochs=args.epochs, batch_size=args.batch_size)
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=eval_input_fn(args.test_data, feature_stats, batch_size=args.batch_size),
            steps=None,
            throttle_secs=25,
        ),
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--job-dir', type=str, required=True)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--test-data', type=str, required=True)
    parser.add_argument('--feature-stats', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--embedding-size', type=int, default=16)
    parser.add_argument('--encoder-residual-layers', type=int, default=1)
    parser.add_argument('--decoder-layers', type=int, default=3)
    parser.add_argument('--previous-output-sampling-p', type=float, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--reg-scale', type=float, default=1e-3)

    main(parser.parse_args())
