from tensorflow import cast, float32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
from typing import Any
import os


# Class that handles data and prepares it for model
class Dataset:
    BUFFER_SIZE = 10000
    BATCH_SIZE = 1000
    VALIDATION_PERCENTAGE = 0.1

    def __init__(self) -> None:
        # EMNIST dataset is used because it is one of the best
        # options available to train the model for VIN recognition
        emnist = tfds.load('emnist', as_supervised=True)
        train_validation, test = emnist['train'], emnist['test']
        self._train, self._validation, self._test = self._prepare(train_validation, test)

    @property
    def train(self) -> Any:
        return self._train

    @property
    def validation(self) -> Any:
        return self._validation

    @property
    def test(self) -> Any:
        return self._test

    @staticmethod
    def _scale(image, label) -> tuple:
        image = cast(image, float32)
        return image / 255., label

    def _prepare(self, init_train_validation, init_test) -> tuple:
        # Scale and shuffle datasets
        def process(sample):
            scaled = sample.map(Dataset._scale)
            return scaled.shuffle(self.BUFFER_SIZE)
        
        train_validation_data = process(init_train_validation)
        test_data = process(init_test)
        
        valiadion_count = int(min(1, max(0, self.VALIDATION_PERCENTAGE)) * len(train_validation_data))
        validation_data = train_validation_data.take(valiadion_count)
        train_data = train_validation_data.skip(valiadion_count)
        
        train_data = train_data.batch(self.BATCH_SIZE)
        validation_data = validation_data.batch(valiadion_count)
        test_data = test_data.batch(len(test_data))
        
        v_inputs, v_targets = next(iter(validation_data))

        return train_data, (v_inputs, v_targets), test_data


# Expects images 28x28 pixels
class Model:
    NUM_EPOCHS = 12

    def __init__(self) -> None:
        self._model = Sequential([
            Flatten(input_shape=(28, 28, 1)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(62, activation='softmax')
        ])
        self._model.compile(optimizer='adam', 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy'])
    
    def fit(self, dataset: Dataset) -> None:
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        self._model.fit(dataset.train, 
                        epochs=self.NUM_EPOCHS, 
                        verbose=2, 
                        batch_size=dataset.BATCH_SIZE,
                        validation_data=dataset.validation,
                        callbacks=[early_stopping])
        loss, accuracy = self._model.evaluate(dataset.test)
        print(f'Test loss: {loss}')
        print(f'Test accuracy: {accuracy}')
    
    def save(self) -> None:
        file_path = os.path.abspath(__file__)
        path = os.path.dirname(file_path) + '/model.h5'
        self._model.save(path)


def generate_model() -> None:
    dataset = Dataset()
    model = Model()
    model.fit(dataset)
    model.save()


if __name__ == '__main__':
    generate_model()