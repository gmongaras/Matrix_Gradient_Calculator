from classes.Matrix import Matrix



class Transpose:
    def __init__(self, matrix, name=None):
        self.name = name
        self.matrix = matrix
        
    def get_grad(self, prev_grad):
        return Transpose(prev_grad)
    
    def __str__(self, ):
        return f"{self.matrix}^T" if isinstance(self.matrix, Matrix) else f"({self.matrix})^T"




class Matmul:
    def __init__(self, left=None, right=None, name=None):
        self.name = name
        self.left = left
        self.right = right
        
    def get_grad_wrt_left(self, prev_grad):
        return Matmul(prev_grad, Transpose(self.right))
    
    def get_grad_wrt_right(self, prev_grad):
        return Matmul(Transpose(self.left), prev_grad)
    
    def __str__(self,):
        return f"{self.left} {self.right}"
    
    
class Hadamard:
    def __init__(self, left=None, right=None, name=None):
        self.name = name
        self.left = left
        self.right = right
        
    def get_grad_wrt_left(self, prev_grad):
        return Hadamard(prev_grad, self.right)
    
    def get_grad_wrt_right(self, prev_grad):
        return Hadamard(self.left, prev_grad)
    
    def __str__(self,):
        left_ = str(self.left) if isinstance(self.left, Matrix) else f"[{self.left}]"
        right_ = str(self.right) if isinstance(self.right, Matrix) else f"[{self.right}]"
        return f"({left_} * {right_})"