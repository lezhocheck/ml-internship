import os
import glob
import numpy as np
from typing import Any
import argparse
from tensorflow.keras.models import load_model
from tensorflow.data import Dataset
from tensorflow.io import read_file
from tensorflow.image import decode_image, resize
from tensorflow import cast, float32


class Data:
    EXT = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    BATCH_SIZE = 1000

    def __init__(self, folder: str) -> None:
        self._folder = os.path.abspath(folder)
        self._prepared = self._prepare_images()

    @property
    def images(self) -> tuple:
        return self._prepared

    def _find_images(self) -> list:
        images = []
        for ext in self.EXT:
            path = f'{self._folder}/{ext}'
            images.extend(glob.glob(path))
        return images
    
    def _prepare_images(self) -> tuple:
        paths = self._find_images()
        def preprocess(path):
            image_contents = read_file(path)
            image = decode_image(image_contents, channels=1)
            image = resize(image, [28, 28])
            # EMNIST images are rotated and flipped.
            # So, if test images in the folder are in right orientation, 
            # need to transform them to match the training data
            image = np.rot90(image, k=-1)
            image = np.fliplr(image)
            image = cast(image, float32)
            image /= 255.
            return image
        images = list(map(lambda x: preprocess(x), paths))
        image_tensor = Dataset.from_tensor_slices(images)
        return image_tensor.batch(self.BATCH_SIZE), paths


class Model:
    def __init__(self) -> None:
        path = os.path.abspath(__file__)
        model_path = os.path.join(os.path.dirname(path), 'model.h5')
        self.model = load_model(model_path)

    def predict(self, data: Data) -> list:
        test, paths = data.images
        predictions = self.model.predict(test)
        def to_ascii(label: int) -> int:
            if label < 10:
                return 48 + label
            elif label < 36:
                return label + 55
            return label + 61

        results = np.argmax(predictions, axis=1)
        formatted = []
        for ind, x in enumerate(results):
            formatted.append(f'{"{:03d}".format(to_ascii(x))}, {paths[ind]}')
        return formatted


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the folder with images.')
    args = parser.parse_args()
    data = Data(args.input)
    model = Model()
    results = model.predict(data)
    print('\n'.join(results))


if __name__ == '__main__':
    run()
