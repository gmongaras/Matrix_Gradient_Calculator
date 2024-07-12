from classes.Matrix import Matrix
from copy import deepcopy



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
    
    def copy(self,):
        return Transpose(
            self.matrix.copy(),
            self.name
        )
    
    
    
    
class Power:
    def __init__(self, matrix, power, name=None):
        self.name = name
        self.matrix = matrix
        self.power = power
        
    # Let O = A B^p
    # Then the gradient wrt B is: \sum_{i=0}^{p-n-1}[(B^T)^n dL A^T (B^T)^(p-1)]
    def get_grad_wrt_right(self, prev_grad, other_side, operation):
        if operation == "matmul":
            # equation = f"({self.matrix}^T)^i ({other_side})^T ({prev_grad}) ({self.matrix}^T)^({self.power} - 1 - i)"
            left = Power(Transpose(self.matrix), f"i")
            right = Power(Transpose(self.matrix), "({self.power} - 1 - i)")
            equation = Matmul(Matmul(Matmul(left, Transpose(other_side)), prev_grad), right)
            return Summation(0, self.power - 1, equation, "i")
        elif operation == "hadamard":
            raise NotImplementedError
        else:
            raise ValueError
    
    # Let O = B^p A
    # Then the gradient wrt B is: \sum_{i=0}^{p-1}[(B^T)^n dL A^T (B^T)^(p-n-1)]
    def get_grad_wrt_left(self, prev_grad, other_side, operation):
        if operation == "matmul":
            # equation = f"({self.matrix}^T)^({self.power} - 1 - i) ({prev_grad}) ({other_side})^T ({self.matrix}^T)^i"
            left = Power(Transpose(self.matrix), f"({self.power} - 1 - i)")
            right = Power(Transpose(self.matrix), "i")
            equation = Matmul(Matmul(Matmul(left, prev_grad), Transpose(other_side)), right)
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
    
    def copy(self):
        return Power(
            self.matrix.copy(),
            self.power,
            self.name
        )
 
 
    
    
# Kind of a dummy class for the output for now
class Summation:
    def __init__(self, lower, upper, equation, index, name=None):
        self.lower = lower
        self.upper = upper
        self.equation = equation
        self.index = index
        self.name = name
        
    def get_grad(self, prev_grad):
        raise NotImplementedError
    
    def __str__(self, ):
        return f"[Summation from {self.lower} to {self.upper} of {self.equation} wrt {self.index}]"
    
    def simulate(cur_shape):
        raise NotImplementedError
    
    def copy(self,):
        return Summation(
            self.lower.copy() if not isinstance(self.lower, str) else self.lower,
            self.upper.copy() if not isinstance(self.upper, str) else self.upper,
            self.equation.copy() if not isinstance(self.equation, str) else self.equation,
            self.index.copy() if not isinstance(self.index, str) else self.index,
            self.name
        )




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
    
    def copy(self,):
        return Matmul(
            self.left.copy() if self.left is not None else None,
            self.right.copy() if self.right is not None else None,
            self.name
        )
    
    
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
    
    def copy(self,):
        return Hadamard(
            self.left.copy() if self.left is not None else None,
            self.right.copy() if self.right is not None else None,
            self.name
        )
    
    
class Add:
    def __init__(self, left=None, right=None, name=None):
        self.name = name
        self.left = left
        self.right = right
        
    def get_grad_wrt_left(self, prev_grad):
        return prev_grad
    
    def get_grad_wrt_right(self, prev_grad):
        return prev_grad
    
    def __str__(self,):
        left_ = str(self.left) if isinstance(self.left, Matrix) else f"[{self.left}]"
        right_ = str(self.right) if isinstance(self.right, Matrix) else f"[{self.right}]"
        return f"({left_} + {right_})"
    
    def simulate(cur_shape, mat_shape):
        assert len(cur_shape) == len(mat_shape)
        assert cur_shape == mat_shape
        return cur_shape
    
    def copy(self,):
        return Add(
            self.left.copy() if self.left is not None else None,
            self.right.copy() if self.right is not None else None,
            self.name
        )
    
    
    
    
    
# Function applied to a matrix such as ReLU, Sigmoid, Softmax, etc.
class MatrixFunction:
    def __init__(self, matrix, name):
        self.name = name
        self.matrix = matrix
        
    # Gradient is the Hadamard product of the gradient of the function and the previous gradient
    def get_grad(self, prev_grad):
        return Hadamard(MatrixFunctionGrad(self.matrix, self.name), prev_grad)
    
    def __str__(self, ):
        return f"{self.name}({self.matrix})"
    
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return MatrixFunction(
            self.matrix.copy(),
            self.name
        )
    
# Dummy class for the gradient of a matrix function
class MatrixFunctionGrad:
    def __init__(self, matrix, name):
        self.name = name
        self.matrix = matrix
        
    def get_grad(self, prev_grad):
        raise NotImplementedError
    
    def __str__(self, ):
        return f"{self.name}'({self.matrix})"
    
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return MatrixFunctionGrad(
            self.matrix.copy(),
            self.name
        )