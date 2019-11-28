import numpy as np

class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)
    
    @staticmethod
    def reLU(arr):
        arr1 = np.where(arr <= 0, 0, 1)
        return arr1
