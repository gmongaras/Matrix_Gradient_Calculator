from copy import deepcopy

class Matrix:
    def __init__(self, shape, label):
        assert isinstance(shape, list)
        assert isinstance(label, str)
        
        self.shape = shape
        self.label = label
        self.grad = []
    
    def add_grad(self, new_grad):
        self.grad.append(new_grad)
        
    def __str__(self, ):
        return self.label
    
    def torch_str(self, ):
        return self.label
    
    def copy(self,):
        return Matrix(
            deepcopy(self.shape),
            self.label
        )