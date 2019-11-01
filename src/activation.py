import numpy as np
class Activation:

    @staticmethod
    def sigmoid( x):
        return 1 / (1 + np.e ** -x)
    
    @staticmethod
    def reLU(x):
        return np.maximum(0.0, x)
