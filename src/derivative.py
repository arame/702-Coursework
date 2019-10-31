

class Derivative:
    def sigmoid(self, x):
        return x * (1.0 - x)
    
    def reLU(self, arr):
        for i in range(len(arr)):
            if arr[i] <= 0:
                arr[i] = 0
            else:
                arr[i] = 1
        return arr
    derivative_function = reLU