import numpy as np

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
            return self.__call__(x)
        else:
            return self.__call__(self.prev.call(x))

class Softmax(object):

    def softmax(self, x):
        exp = np.exp(x)
        return x / sum(exp)

    def __call__(self, x):
        return self.softmax(x)

    def grad(self, x):
        pass

class Sigmoid(object):

    def sigmoid(self, x):
        exp = np.exp(x)
        return 1 / (1 + 1 / exp)

    def __call__(self, x):
        return self.sigmoid(x)

class ReLU(object):

    def relu(self, x):
        return x * (x > 0)

    def __call__(self, x):
        return self.relu(x)
