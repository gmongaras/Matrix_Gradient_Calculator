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
    
    def torch_str(self,):
        if isinstance(self.matrix, Matrix):
            return f"{self.matrix.torch_str()}.mT"
        else:
            return f"({self.matrix.torch_str()}).mT"
    
    @staticmethod
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
        
    @staticmethod
    def simplify(grad):
        # Get the matrix that is being transposed
        matrix = grad.matrix
        
        # If this is a matrix, we don't have to worry about it
        if isinstance(matrix, Matrix):
            return grad
        
        # If this is a transpose, we can remove the transpose
        elif isinstance(matrix, Transpose):
            grad = matrix.matrix
            
        # If this is a power, we can move the transpose into the power
        elif isinstance(matrix, Power):
            matrix.matrix = Transpose(matrix.matrix)
            grad = matrix
            
        # If this is a matmul, we can reverse the order of the matmul
        # and transpose each matrix
        elif isinstance(matrix, Matmul):
            grad = matrix
            tmp = grad.left
            grad.left = Transpose(grad.right.copy())
            grad.right = Transpose(tmp.copy())
        
        # If this is a hadamard or Add, we can transpose each matrix
        elif isinstance(matrix, Hadamard) or isinstance(matrix, Add):
            grad = matrix
            grad.left = Transpose(grad.left)
            grad.right = Transpose(grad.right)
            
        # If this is a matrix function grad, we can transpose the matrix
        elif isinstance(matrix, MatrixFunctionGrad) or isinstance(matrix, MatrixFunction):
            grad = matrix
            grad.matrix = Transpose(grad.matrix)
            
        # If this is a matrix vector function grad
        elif isinstance(matrix, Jacobian) or isinstance(matrix, MatrixVectorFunction):
            # If it's a vector function, we can transpose the matrices and switch the dimensions
            if isinstance(matrix, MatrixVectorFunction):
                grad = matrix
                if grad.dim == -1:
                    grad.dim = -2
                else:
                    grad.dim = -1
                grad.matrix = Transpose(grad.matrix)
            # TODO: FOr now we will just transpose the Jacobian. However it may be possible to simplify
            else:
                return grad
                    
            
        # If this is a summation, we can transpose the equation
        elif isinstance(matrix, Summation):
            grad = matrix
            grad.equation = Transpose(grad.equation)
        
        return grad
    
    
    
    
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

    def torch_str(self,):
        if isinstance(self.matrix, Matrix):
            return f"{self.matrix.torch_str()}.matrix_power({self.power})"
        else:
            return f"{self.matrix.torch_str()}.matrix_power({self.power})"
    
    @staticmethod
    def simulate(cur_shape):
        assert len(cur_shape) >= 2
        return cur_shape
    
    def copy(self):
        return Power(
            self.matrix.copy(),
            self.power,
            self.name
        )
        
    @staticmethod
    def simplify(grad):
        return grad
 
 
    
    
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
    
    def torch_str(self,):
        return f"[Summation from {self.lower} to {self.upper} of {self.equation.torch_str()} wrt {self.index}]"
    
    @staticmethod
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
        
    @staticmethod
    def simplify(grad):
        return grad




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
    
    def torch_str(self,):
        # For Jacobian, we need to do a three-way matrix multiplication. This
        # can be done with a einsum operation
        grad_type = "normal"
        if isinstance(self.left, Jacobian)  or (isinstance(self.left, Transpose) and isinstance(self.left.matrix, Jacobian)):
            grad_type = "jacobian_left"
        elif isinstance(self.right, Jacobian) or (isinstance(self.right, Transpose) and isinstance(self.right.matrix, Jacobian)):
            grad_type = "jacobian_right"
            
        # Special grad types
        if grad_type == "jacobian_left":
            # If the Jacobian is Transposed, I am just going to untranspose it
            if isinstance(self.left, Transpose):
                self.left = self.left.matrix
            left_str = "ijk"
            # If the right is transposed, we need to transpose the right
            right_str = "ik"
            # If on the left, this isn't transposed
            end_str = "ij"
            return f"torch.einsum(\"...{left_str},...{right_str}->...{end_str}\", {self.left.torch_str()}, {self.right.torch_str()})"
        elif grad_type == "jacobian_right":
            # Same as above
            if isinstance(self.right, Transpose):
                self.right = self.right.matrix
            left_str = "ijk"
            right_str = "ki"
            # If on the right, this is transposed
            end_str = "ji"
            return f"torch.einsum(\"...{left_str},...{right_str}->...{end_str}\", {self.right.torch_str()}, {self.left.torch_str()})"
        
                
        # Otherwise it's just normal
        return f"{self.left.torch_str()} @ {self.right.torch_str()}"
    
    @staticmethod
    def simulate(cur_shape, mat_shape):
        assert len(cur_shape) >= 2
        # assert len(cur_shape) == len(mat_shape)
        assert cur_shape[-1] == mat_shape[-2]
        cur_shape[-1] = mat_shape[-1]
        return cur_shape
    
    def copy(self,):
        return Matmul(
            self.left.copy() if self.left is not None else None,
            self.right.copy() if self.right is not None else None,
            self.name,
        )
        
    @staticmethod
    def simplify(grad):
        return grad
    
    
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
    
    def torch_str(self,):
        left_ = self.left.torch_str() if isinstance(self.left, Matrix) else f"({self.left.torch_str()})"
        right_ = self.right.torch_str() if isinstance(self.right, Matrix) else f"({self.right.torch_str()})"
        return f"({left_} * {right_})"
    
    @staticmethod
    def simulate(cur_shape, mat_shape):
        assert len(cur_shape) >= 2
        assert len(cur_shape) == len(mat_shape)
        assert cur_shape == mat_shape
        return cur_shape
    
    def copy(self,):
        return Hadamard(
            self.left.copy() if self.left is not None else None,
            self.right.copy() if self.right is not None else None,
            self.name
        )
        
    @staticmethod
    def simplify(grad):
        return grad
    
    
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
    
    def torch_str(self,):
        left_ = self.left.torch_str() if isinstance(self.left, Matrix) else f"({self.left.torch_str()})"
        right_ = self.right.torch_str() if isinstance(self.right, Matrix) else f"({self.right.torch_str()})"
        return f"({left_} + {right_})"
    
    @staticmethod
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
        
    @staticmethod
    def simplify(grad):
        return grad
    
    
    
    
    
# Function applied to a matrix such as ReLU, Sigmoid, etc.
# These functions are applied scalar-wise. Not row-wise.
class MatrixFunction:
    def __init__(self, matrix, name):
        self.name = name
        self.matrix = matrix
        
    # Gradient is the Hadamard product of the gradient of the function and the previous gradient
    def get_grad(self, prev_grad):
        return Hadamard(MatrixFunctionGrad(self.matrix, self.name), prev_grad)
    
    def __str__(self, ):
        return f"{self.name}({self.matrix})"
    
    def torch_str(self,):
        return f"{self.name}({self.matrix.torch_str()})"
    
    @staticmethod
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return MatrixFunction(
            self.matrix.copy(),
            self.name
        )
    
    @staticmethod
    def simplify(grad):
        return grad
# Dummy class for the gradient of a matrix function
class MatrixFunctionGrad:
    def __init__(self, matrix, name):
        self.name = name
        self.matrix = matrix
        
    def get_grad(self, prev_grad):
        raise NotImplementedError
    
    def __str__(self, ):
        return f"{self.name}'({self.matrix})"
    
    def torch_str(self,):
        return f"{self.name}_der({self.matrix.torch_str()})"
    
    @staticmethod
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return MatrixFunctionGrad(
            self.matrix.copy(),
            self.name
        )
        
    @staticmethod
    def simplify(grad):
        return grad
        
        
        
# Function applied to a matrix such Softmax.
# These functions are applied row-wise.
class MatrixVectorFunction:
    def __init__(self, matrix, name, dim=-1):
        self.name = name
        self.matrix = matrix
        
        # Assuming the dimension is over the last axis
        self.dim = dim
        
    # Gradient is the Matmul of the gradient of the function and the previous gradient
    def get_grad(self, prev_grad):
        return Matmul(Jacobian(MatrixVectorFunction(self.matrix, self.name, dim=self.dim)), prev_grad)
    
    def __str__(self, ):
        return f"{self.name}({self.matrix}, dim={self.dim})"
    
    def torch_str(self,):
        return f"{self.name}({self.matrix.torch_str()}, dim={self.dim})"
    
    @staticmethod
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return MatrixVectorFunction(
            self.matrix.copy(),
            self.name,
            self.dim
        )
        
    @staticmethod
    def simplify(grad):
        return grad
# Dummy class for the gradient of a vector function which is the Jacobian
class Jacobian:
    def __init__(self, matrix, name=None):
        self.name = name
        self.matrix = matrix
        
    def get_grad(self, prev_grad):
        raise NotImplementedError
    
    def __str__(self, ):
        return f"Jacobian({self.matrix})"
    
    def torch_str(self,):
        return f"Jacobian({self.matrix.torch_str()})"
    
    @staticmethod
    def simulate(cur_shape):
        return cur_shape
    
    def copy(self,):
        return Jacobian(
            self.matrix.copy(),
            self.name
        )
        
    @staticmethod
    def simplify(grad):
        return grad