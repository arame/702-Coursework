import numpy as np
class Activation:

    def sigmoid(self, x):
        return 1 / (1 + np.e ** -x)
    
    activation_function = sigmoid