import numpy as np
class Activation:

    def sigmoid(self, x):
        return 1 / (1 + np.e ** -x)
    
    def reLU(self, x):
        return np.maximum(0.0, x)

    activation_function = reLU