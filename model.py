#from keras.layers import Input, LSTM, Dense, Embedding, Merge, Bidirectional
#from keras.models import model_from_json, Sequential
import logging
import globals
import numpy as np
from activations import Activation
from initializers import uniform_initializer
from loss import BinaryCrossentropy, MSE

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class Model(object):

    def __init__(self):
        self.layers = []

    def add(self, layer):
        if isinstance(layer, Layer):
            layer.build()

        if (self.layers == []):
            self.layers.append(layer)
        else:
            last_layer = self.layers[-1]
            layer.prev = last_layer

            if isinstance(layer, Activation):
                layer.input_dim = last_layer.output_dim
            self.layers.append(layer)


    def compile(self, loss):
        if loss == "binary_crossentropy":
            self.loss = BinaryCrossentropy(self)
        elif loss == "mse":
            self.loss = MSE(self)
        else:
            logger.warning("Invalid loss function.")
            self.loss = None


    def fit(self,
            X, Y,
            batch_size=32,
            iterations=500,
            validation_data=None):
        history = []
        batches = range(N / batch_size)
        batches = [(i * batch_size, (i+1) * batch_size) for i in batches]
        for epoch in xrange(iterations):
            N, _ = X.shape
            np.random.shuffle(batches)
            total_loss = 0
            for start, end in batches:
                batchX = X[start:end]
                batchY = Y[start:end]
                loss = self.loss.apply_grad(batchX, batchY)
                total_loss += loss

                if validation_data is not None:
                    pass
            history.append((epoch, total_loss))
        return history


    def predict(self, X):
        n, input_dim = X.shape
        if len(self.layers) == 0:
            logger.warning("Model has no layers.")
            return None

        if input_dim != self.layers[0].input_dim:
            logger.warning("Input dimension does not match.")
            return None

        return self.layers[-1].call(X)


class Layer(object):

    def __init__(self,
                 output_dim,
                 input_dim = None,
                 initializer = uniform_initializer):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.initializer = initializer
        self.prev = None
        self.next = None
        #self.activation = getattr(activations, activation)
        #self.W_regularizer = W_regularizer

    def build(self):
        self.W = self.initializer((self.input_dim, self.output_dim))
        self.b = np.zeros((1, self.output_dim))

    def __call__(self, X):
        return np.dot(X, self.W) + self.b

    def call(self, x):
        if self.prev is None:
            return self.__call__(x)
        else:
            return self.__call__(self.prev.call(x))

