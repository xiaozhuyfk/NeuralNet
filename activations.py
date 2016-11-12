import numpy as np
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def softmax(x):
    return Softmax()(x)

def sigmoid(x):
    return Sigmoid()(x)

def relu(x):
    return ReLU()(x)

class Activation(object):

    def __init__(self, activation):
        self.prev = None
        self.next = None
        self.W = None
        self.b = None
        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "relu":
            self.activation = ReLU()
        else:
            logger.warning("Invalid activation input.")

    def __call__(self, x):
        return self.activation(x)

    def call(self, x):
        if self.prev is None:
            self.output = self.__call__(x)
            return self.output
        else:
            self.output = self.__call__(self.prev.call(x))
            return self.output

    def apply_grad(self, X, Y, gradients):
        assert(self.prev is not None)

        #output = self.prev.call(X)
        output = self.prev.output
        for i in xrange(len(gradients)):
            oi = output[i:i+1]
            gradients[i] *= self.activation.grad(oi)
        self.prev.apply_grad(X, Y, gradients)

class Softmax(object):

    def softmax(self, x):
        exp = np.exp(x)
        return exp / exp.sum(axis=1)

    def __call__(self, x):
        return self.softmax(x)

    def grad(self, x):
        return NotImplemented

class Sigmoid(object):

    def sigmoid(self, x):
        exp = np.exp(x)
        return 1 / (1 + 1 / exp)

    def __call__(self, x):
        return self.sigmoid(x)

    def grad(self, x):
        return self(x) * (1 - self(x))

class ReLU(object):

    def relu(self, x):
        return x * (x > 0)

    def __call__(self, x):
        return self.relu(x)

    def grad(self, x):
        return (x > 0) * 1.0
