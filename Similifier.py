from classes.Matrix import Matrix
from classes.Operations import *





class Similifier:
    def __init__(self):
        pass
    
    
    
    # Used to simplify a transpose operation
    def simplify_transpose(self, matrices_and_functions, grad):
        return Transpose.simplify(grad)
            
            
    # Used to simplify a matmul operation
    def simplify_matmul(self, matrices_and_functions, grad):
        return Matmul.simplify(grad)
    
    
    
    # Used to simplify a hadamard operation
    def simplify_hadamard(self, matrices_and_functions, grad):
        return Hadamard.simplify(grad)
    
    
    
    # Used to simplify a summation operation
    def simplify_summation(self, matrices_and_functions, grad):
        return Summation.simplify(grad)
            
            
            
    def simplify_grad(self, matrices_and_functions, grad):
        # Matrices cannot be similified obviously
        if isinstance(grad, Matrix):
            return grad
        
        # If the function is a transpose
        elif isinstance(grad, Transpose):
            # Matrices and Jacobian functions cannot be simplified
            if isinstance(grad.matrix, Matrix) or isinstance(grad.matrix, Jacobian):
                return grad
            grad = self.simplify_transpose(matrices_and_functions, grad)
            grad = self.simplify_grad(matrices_and_functions, grad)
            
        # If the function is a power
        elif isinstance(grad, Power):
            # Simplify the matrix
            grad.matrix = self.simplify_grad(matrices_and_functions, grad.matrix)
            
        # If the function is a Matmul
        elif isinstance(grad, Matmul):
            # Simplify left and right, then simplify the matmul
            grad.left = self.simplify_grad(matrices_and_functions, grad.left)
            grad.right = self.simplify_grad(matrices_and_functions, grad.right)
            grad = self.simplify_matmul(matrices_and_functions, grad)
            
        # If the function is a Hadamard or Add
        elif isinstance(grad, Hadamard) or isinstance(grad, Add):
            # Simplify left and right
            grad.left = self.simplify_grad(matrices_and_functions, grad.left)
            grad.right = self.simplify_grad(matrices_and_functions, grad.right)
            
        # If the function is a MatrixFunctionGrad
        elif isinstance(grad, MatrixFunctionGrad) or isinstance(grad, MatrixFunction):
            # Simplify the matrix
            grad.matrix = self.simplify_grad(matrices_and_functions, grad.matrix)
            
        # If the function is a MatrixVectorFunction
        elif isinstance(grad, MatrixVectorFunction):
            # Simplify the matrix
            grad.matrix = self.simplify_grad(matrices_and_functions, grad.matrix)
            
        # For now we do not simplify Jacobian functions
        elif isinstance(grad, Jacobian):
            return grad
            
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
            new_grads = []
            for grad in function:
                grad = self.simplify_grad(matrices_and_functions, grad)
                new_grads.append(grad)
            matrices_and_functions[matrix].grad = new_grads