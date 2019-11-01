

class Derivative:
    @staticmethod
    def sigmoid(x):
        return x * (1.0 - x)
    
    @staticmethod
    def reLU(arr):
        for i in range(len(arr)):
            if arr[i] <= 0:
                arr[i] = 0
            else:
                arr[i] = 1
        return arr
