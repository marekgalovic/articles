import os
from argparse import ArgumentParser
from PIL import Image
import json

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex


def open_img(path):
    img = Image.open(path)
    return img.resize((224, 224))


def load_model():
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.image.convert_image_dtype(x, tf.float32)),
        hub.KerasLayer("https://tfhub.dev/tensorflow/resnet_50/feature_vector/1", trainable=False)  
    ])


def get_feature_vectors(model, images):
    images = np.stack(list(map(lambda img: np.asarray(img), images)))
    
    return model(images)


def main(args):
    index = AnnoyIndex(2048, 'euclidean')
    index_metadata = {}

    model = load_model()

    batch = []
    total_size = 0
    for i, fname in enumerate(os.listdir(args.images_dir)):
        if not (fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.jpeg')):
            continue

        path = os.path.join(args.images_dir, fname)
        try:
            img = open_img(path)
            batch.append((i, img, fname))
        except Exception as e:
            print(e)
            continue

        if len(batch) == args.batch_size:
            total_size += len(batch)
            print("Process batch: %d" % total_size)
            ids, imgs, img_fnames = zip(*batch)
            vectors = get_feature_vectors(model, imgs).numpy()
            for j, vector in enumerate(vectors):
                index.add_item(ids[j], vector.tolist())
                index_metadata[ids[j]] = {
                    'filename': img_fnames[j]
                }

            batch = []

            if total_size >= args.max_items:
                break

    print('Build index')
    index.build(args.n_trees)
    print('Save index')
    index.save(os.path.join(args.dst, 'index.ann'))
    json.dump(index_metadata, open(os.path.join(args.dst, 'index_metadata.json'), 'w'))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--dst', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-trees', type=int, default=10)
    parser.add_argument('--max-items', type=int, default=10000)

    main(parser.parse_args())
