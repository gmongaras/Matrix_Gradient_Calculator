from classes.Matrix import Matrix
from classes.Operations import Transpose, Power, Matmul, Hadamard, MatrixFunctionGrad, Summation





class Similifier:
    def __init__(self):
        pass
    
    
    
    # Used to simplify a transpose operation
    def simplify_transpose(self, matrices_and_functions, grad):
        # Get the matrix that is being transposed
        matrix = grad.matrix
        
        # If this is a matrix, we don't have to worry about it
        if isinstance(matrix, Matrix):
            return grad
        
        # If this is a transpose, we can remove the transpose
        if isinstance(matrix, Transpose):
            # matrices_and_functions[matrix.matrix.label] = matrix.matrix
            # del matrices_and_functions[matrix.label]
            # del matrices_and_functions[function.label]
            grad = matrix.matrix
        
        return grad
            
            
    # Used to simplify a matmul operation
    def simplify_matmul(self, matrices_and_functions, grad):
        # We can simplify a matmul to a power if the matrices are the same
        if grad.left == grad.right:
            matrices_and_functions[grad.left.label] = Power(grad.left, 2)
            del matrices_and_functions[grad.label]
            del matrices_and_functions[grad.left.label]
            del matrices_and_functions[grad.right.label]
            
        return grad
            
            
            
    def simplify_grad(self, matrices_and_functions, grad):
        # Matrices cannot be similified obviously
        if isinstance(grad, Matrix):
            return grad
        
        # If the function is a transpose, we may be able to simplify it
        elif isinstance(grad, Transpose):
            grad.matrix = self.simplify_grad(matrices_and_functions, grad.matrix)
            grad = self.simplify_transpose(matrices_and_functions, grad)
            
        # If the function is a Matmul
        elif isinstance(grad, Matmul):
            # Simplify left and right, then simplify the matmul
            grad.left = self.simplify_grad(matrices_and_functions, grad.left)
            grad.right = self.simplify_grad(matrices_and_functions, grad.right)
            grad = self.simplify_matmul(matrices_and_functions, grad)
            
        # If the function is a Hadamard
        elif isinstance(grad, Hadamard):
            # Simplify left and right
            grad.left = self.simplify_grad(matrices_and_functions, grad.left)
            grad.right = self.simplify_grad(matrices_and_functions, grad.right)
            
        # If the function is a MatrixFunctionGrad
        elif isinstance(grad, MatrixFunctionGrad):
            # Simplify the matrix
            grad.matrix = self.simplify_grad(matrices_and_functions, grad.matrix)
            
        # If the function is a Summation
        elif isinstance(grad, Summation):
            # Simplify the equation
            grad.equation = self.simplify_grad(matrices_and_functions, grad.equation)
            
        return grad

    
    def simplify(self, matrices_and_functions):
        # Get all gradient functions
        functions = {i:j.grad for i, j in matrices_and_functions.items() if isinstance(j, Matrix)}
        
        # Iterate through all the function gradients
        for matrix, function in functions.items():
            # Iterate through all the gradient functions
            for grad in function:
                self.simplify_grad(matrices_and_functions, grad)