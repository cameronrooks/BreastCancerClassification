# BreastCancerClassification

This project uses machine learning to train a model to recognize breast cancer.

The model structure consists of 5 convolutional filters, and 2 linear layers, with batch normalization and dropout utilized between each layer and the ReLU activation function.
The output layer consists of a single node and uses the sigmoid activation function.

The data used to train the model consists of 277 thousand patches of size 50x50. The dataset can be downloaded here: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

After training for around 60 epochs with a learning rate of .01 and batch size of 512, the model converged nicely. 

To train a model yourself, download the data from the above kaggle link, and change the paths in config.py to correspond to where your data is stored. Then, run the following command:
python driver.py train

To generate plots for a trained model, run the following command:
python driver.py plot
