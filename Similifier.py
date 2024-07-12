from classes.Matrix import Matrix
from classes.Operations import Transpose, Power, Matmul, Hadamard, Add, MatrixFunction, MatrixFunctionGrad, Summation





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
            
        # If this is a summation, we can transpose the equation
        elif isinstance(matrix, Summation):
            grad = matrix
            grad.equation = Transpose(grad.equation)
        
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
    
    
    
    # Used to simplify a hadamard operation
    def simplify_hadamard(self, matrices_and_functions, grad):
        return grad
    
    
    
    # Used to simplify a summation operation
    def simplify_summation(self, matrices_and_functions, grad):
        return grad
            
            
            
    def simplify_grad(self, matrices_and_functions, grad):
        # Matrices cannot be similified obviously
        if isinstance(grad, Matrix):
            return grad
        
        # If the function is a transpose
        elif isinstance(grad, Transpose):
            if isinstance(grad.matrix, Matrix):
                return grad
            grad = self.simplify_transpose(matrices_and_functions, grad)
            grad = self.simplify_grad(matrices_and_functions, grad)
            
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