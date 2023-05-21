# CHI ML Internship 2023

## Authors

Oleh Toporkov: 
- oleh.toporkov@gmail.com
- [GitHub]((https://github.com/lezhocheck))
- [Linkedin](https://www.linkedin.com/in/lezhocheck/)

---

## Solution

### Dataset

To classify grayscale images (characters of VIN) the extended MNIST dataset (https://arxiv.org/pdf/1702.05373v1.pdf) was choosen.

The EMNIST dataset is a good option for training a model that needs to classify VIN numbers due to several reasons. First, the dataset contains a large and diverse collection of handwritten characters, including both uppercase and lowercase letters, as well as digits. This diversity aligns well with the variety of characters found in VIN numbers, which consist of both letters and digits.

Second, the EMNIST dataset provides a standardized and labeled dataset for character recognition tasks. Each character in the dataset is accurately labeled, allowing for supervised training of the model. This ensures that the model can learn from correctly annotated examples and make accurate predictions on unseen data.

Before training the model, the data is preprocessed by scaling, shuffling, dividing into training, validation and test samples.

### Model

To solve the problem, a supervised deep learning model was built.
The model architecture is defined using the Sequential API from the Tensorflow Keras library. The model consists of several layers:

- Flatten Layer is responsible for flattening the input grayscale images into a 1-dimensional array.

- There are three dense layers with 256, 256, and 128 units, respectively. Each dense layer is followed by the ReLu activation function, which introduces non-linearity to the model.

- The output layer has 62 units, corresponding to the number of classes in the classification task (VINs contain only upper-case letters and digits, but some letters can be written as lower-case, so it is better to classify them as lower-case letters and then convert them to upper-case if necessary). It uses the softmax activation function to produce a probability distribution over the classes.

The model is compiled with the Adam optimizer, which is one of the most efficient stochastic gradient descent algorithms. The loss function used is sparse categorical cross-entropy, suitable for categorical classification problems where the target labels are integers.

### Results

After training the model for 12 epochs, the following results were obtained:
- training accuracy ≈ 0.86
- validation accuracy ≈ 0.85

Also test accuracy was measured:
- test accuracy ≈ 0.85

These test results demonstrate capability of the model to accurately classify VINs in real-world scenarios. The high test accuracy suggests that the model has learned relevant patterns and features from the training data, allowing it to generalize well and perform effectively on unseen VIN images. 

However, further improvements can be made to make the predictions more robust:
- additional hyperparameter tuning, such as learning rate, batch size, and number of layers, can potentially improve its performance.

- trying more sophisticated models. While the current model, with its dense layers, has shown good results, experimenting with more complex models like CNNs could yield even better performance.

- using more advanced data preprocessing techniques. The current preprocessing steps, including flattening the input and normalizing the pixel values, have contributed to the model's accuracy. However, exploring advanced techniques like data augmentation, which can generate additional training samples through transformations like rotation, scaling, or flipping, could further enhance the model's ability to generalize to new VINs.

---

## Usage

To create an environment container, run: 

```docker build -t app .```

To start a container: 

```docker run -it --rm app /bin/bash```

The file with the trained model *model.h5* is stored in the root folder

To start training the model, run inside the container:

```python train.py```

To run the model inference script, run inside the container:

```python inference.py --input path/to/folder```

You can also see some data exploration in *explore.ipynb*.