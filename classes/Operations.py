from classes.Matrix import Matrix



class Transpose:
    def __init__(self, matrix, name=None):
        self.name = name
        self.matrix = matrix
        
    def get_grad(self, prev_grad):
        return Transpose(prev_grad)
    
    def __str__(self, ):
        return f"{self.matrix}^T" if isinstance(self.matrix, Matrix) else f"({self.matrix})^T"
    
    def simulate(cur_shape):
        assert len(cur_shape) >= 2
        tmp = cur_shape[-1]
        cur_shape[-1] = cur_shape[-2]
        cur_shape[-2] = tmp
        return cur_shape
    
    
    
    
class Power:
    def __init__(self, matrix, power, name=None):
        self.name = name
        self.matrix = matrix
        self.power = power
        
    # Let O = A B^p
    # Then the gradient wrt B is: \sum_{i=0}^{p-n-1}[(B^T)^n dL A^T (B^T)^(p-1)]
    def get_grad_wrt_right(self, prev_grad, other_side, operation):
        if operation == "matmul":
            equation = f"({self.matrix}^T)^i ({other_side})^T ({prev_grad}) ({self.matrix}^T)^({self.power} - 1 - i)"
            return Summation(0, self.power - 1, equation, "i")
        elif operation == "hadamard":
            raise NotImplementedError
        else:
            raise ValueError
    
    # Let O = B^p A
    # Then the gradient wrt B is: \sum_{i=0}^{p-1}[(B^T)^n dL A^T (B^T)^(p-n-1)]
    def get_grad_wrt_left(self, prev_grad, other_side, operation):
        if operation == "matmul":
            equation = f"({self.matrix}^T)^({self.power} - 1 - i) ({prev_grad}) ({other_side})^T ({self.matrix}^T)^i"
            return Summation(0, self.power - 1, equation, "i")
        elif operation == "hadamard":
            raise NotImplementedError
        else:
            raise ValueError
    
    def __str__(self, ):
        return f"{self.matrix}^{self.power}" if isinstance(self.matrix, Matrix) else f"({self.matrix})^{self.power}"
    
    def simulate(cur_shape):
        assert len(cur_shape) >= 2
        return cur_shape
 
 
    
    
# Kind of a dummy class for the output for now
class Summation:
    def __init__(self, lower, upper, equation, index, name=None):
        self.lower = lower
        self.upper = upper
        self.equation = equation
        self.index = index
        
    def get_grad(self, prev_grad):
        raise NotImplementedError
    
    def __str__(self, ):
        return f"[Summation from {self.lower} to {self.upper} of {self.equation} wrt {self.index}]"
    
    def simulate(cur_shape):
        raise NotImplementedError




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
    
    def simulate(cur_shape, mat_shape):
        assert len(cur_shape) >= 2
        assert len(cur_shape) == len(mat_shape)
        assert cur_shape[-1] == mat_shape[-2]
        cur_shape[-1] = mat_shape[-1]
        return cur_shape
    
    
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
    
    def simulate(cur_shape, mat_shape):
        assert len(cur_shape) == len(mat_shape)
        assert cur_shape == mat_shape
        return cur_shape