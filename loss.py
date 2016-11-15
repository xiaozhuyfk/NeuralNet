import logging, math, globals
import numpy as np
from activations import softmax

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

class BinaryCrossentropy(object):

    def __init__(self, model):
        self.model = model

    def __call__(self, X, Y):
        N, _ = X.shape
        loss = 0
        lam = globals.lam

        if len(self.model.layers) == 0:
            logger.warning("No loss computed. Empty layers.")
            return 0

        O = self.model.predict(X)
        #O = self.model.output

        for i in xrange(N):
            o = O[i]
            y = Y[i]
            loss += math.log((np.exp(o)).sum()) - np.dot(y, o)
        loss = loss / N

        weight_sum = 0
        for layer in self.model.layers:
            if layer.W is not None:
                weight_sum += np.linalg.norm(layer.W)**2

        return loss + lam * weight_sum / 2

    def apply_grad(self, X, Y):
        N, _ = X.shape
        gradients = []
        output = self.model.predict(X)
        self.model.output = output

        for i in xrange(N):
            yi = Y[i:i+1]
            oi = output[i:i+1]
            loss = 1.0/N * (softmax(oi) - yi)
            gradients.append(loss)
        self.model.layers[-1].apply_grad(X, Y, gradients)

        return self(X, Y)

class MSE(object):

    def __init__(self, model):
        self.model = model

    def __call__(self, X, Y):
        N, _ = X.shape
        loss = 0
        lam = globals.lam

        if len(self.model.layers) == 0:
            logger.warning("No loss computed. Empty layers.")
            return 0

        O = self.model.predict(X)
        loss = np.linalg.norm(O - Y)**2 / N

        weight_sum = 0
        for layer in self.model.layers:
            if layer.W is not None:
                weight_sum += np.linalg.norm(layer.W)**2

        return loss + lam * weight_sum / 2

    def apply_grad(self, X, Y):
        N, _ = X.shape
        gradients = []
        output = self.model.predict(X)
        self.model.output = output

        for i in xrange(N):
            yi = Y[i:i+1]
            oi = output[i:i+1]
            loss = 2 * (oi - yi) / N
            gradients.append(loss)
        self.model.layers[-1].apply_grad(X, Y, gradients)

        return self(X, Y)
