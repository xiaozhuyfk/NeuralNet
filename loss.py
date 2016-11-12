import logging, math, globals
import numpy as np

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
        for i in xrange(N):
            o = O[i]
            y = Y[i]
            loss += math.log((np.exp(o)).sum()) - np.dot(y, o)
        loss = loss / N

        weight_sum = 0
        for layer in self.model.layers:
            weight_sum += np.linalg.norm(layer.W)**2

        return loss + lam * weight_sum / 2

    def apply_grad(self, X, Y):
        pass

class MSE(object):

    def __init__(self, model):
        self.model = model

    def __call__(self, X, Y):
        return 0