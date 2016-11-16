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
        if (self.layers == []):
            if layer.input_dim is None:
                logger.warning("First layer has no input dimension.")
            self.layers.append(layer)
        else:
            last_layer = self.layers[-1]
            layer.prev = last_layer

            if isinstance(layer, Layer):
                layer.input_dim = last_layer.output_dim
            elif isinstance(layer, Activation):
                layer.input_dim = last_layer.output_dim
                layer.output_dim = layer.input_dim
            self.layers.append(layer)

        assert(layer.output_dim is not None)
        assert(layer.input_dim is not None)

        if isinstance(layer, Layer):
            layer.build()


    def compile(self, loss):
        if loss == "binary_crossentropy":
            self.loss = BinaryCrossentropy(self)
        elif loss == "mse":
            self.loss = MSE(self)
        else:
            logger.warning("Invalid loss function.")
            self.loss = None

    def evaluate(self, X, Y):
        if isinstance(self.loss, MSE):
            return self.loss.mse(X, Y)

        N, _ = X.shape
        pred_val = self.predict(X)
        y_answer = np.argmax(Y, axis=1)
        pred_answer = np.argmax(pred_val, axis=1)
        accuracy = 1 - float(np.sum(abs(y_answer - pred_answer))) / N
        return accuracy

    def fit(self,
            X, Y,
            batch_size=32,
            iterations=500,
            validation_data=None):
        N, _ = X.shape
        history = []
        batches = range(N / batch_size)
        batches = [(i * batch_size, (i+1) * batch_size) for i in batches]
        for epoch in xrange(1, iterations+1):
            np.random.shuffle(batches)
            total_loss = 0
            for start, end in batches:
                batchX = X[start:end]
                batchY = Y[start:end]
                loss = self.loss.apply_grad(batchX, batchY)
                total_loss += loss

            train_accuracy = self.evaluate(X, Y)
            if validation_data is not None:
                Xval = validation_data[0]
                yval = validation_data[1]
                val_accuracy = self.evaluate(Xval, yval)
                val_loss = self.loss(Xval, yval)
            else:
                val_accuracy = "NO EVALUATION DATA"
                val_loss = "NO EVALUATION DATA"

            history.append([epoch, total_loss, val_loss, train_accuracy, val_accuracy])

            print '-----------------------'
            print 'Epoch', epoch
            print 'Total Training Loss:', total_loss
            print 'Total Validation Loss', val_loss
            print 'Training Accuracy:', train_accuracy
            print 'Validation Accuracy:', val_accuracy
            print '-----------------------'

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

    def build(self):
        self.W = self.initializer((self.input_dim, self.output_dim))
        self.b = np.zeros((1, self.output_dim))

    def __call__(self, X):
        return np.dot(X, self.W) + self.b

    def call(self, x):
        if self.prev is None:
            self.output = self.__call__(x)
            return self.output
        else:
            self.output = self.__call__(self.prev.call(x))
            return self.output

    def apply_grad(self, X, Y, gradients):
        lam = globals.lam
        alpha = globals.alpha

        W_gradients = []
        b_gradients = []
        if self.prev is None:
            for i in xrange(len(gradients)):
                grad = gradients[i]
                xi = X[i:i+1]
                W_gradients.append(np.outer(xi, grad))
                b_gradients.append(grad)
            self.W -= alpha * (sum(W_gradients) + lam * self.W)
            self.b -= alpha * (sum(b_gradients))
            return

        assert(self.prev is not None)
        #output = self.prev.call(X)
        output = self.prev.output
        for i in xrange(len(gradients)):
            grad = gradients[i]
            oi = output[i:i+1]
            W_gradients.append(np.outer(oi, grad))
            b_gradients.append(grad)
            gradients[i] = np.dot(grad, np.transpose(self.W))

        # update weight matrix
        self.W -= alpha * (sum(W_gradients) + lam * self.W)
        self.b -= alpha * (sum(b_gradients))

        self.prev.apply_grad(X, Y, gradients)




