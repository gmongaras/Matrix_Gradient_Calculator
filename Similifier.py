from classes.Matrix import Matrix
from classes.Operations import Transpose, Power, Matmul, Hadamard





class Similifier:
    def __init__(self):
        pass
    
    
    
    # Used to simplify a transpose operation
    def simplify_transpose(self, matrices_and_functions, grad):
        # Get the matrix that is being transposed
        matrix = grad.matrix
        
        # If this is a matrix, we don't have to worry about it
        if isinstance(matrix, Matrix):
            return
        
        # If this is a transpose, we can remove the transpose
        if isinstance(matrix, Transpose):
            matrices_and_functions[matrix.matrix.label] = matrix.matrix
            del matrices_and_functions[matrix.label]
            del matrices_and_functions[function.label]
            
            
    # Used to simplify a matmul operation
    def simplify_matmul(self, matrices_and_functions, grad):
        # We can simplify a matmul to a power if the matrices are the same
        if grad.left == grad.right:
            matrices_and_functions[grad.left.label] = Power(grad.left, 2)
            del matrices_and_functions[grad.label]
            del matrices_and_functions[grad.left.label]
            del matrices_and_functions[grad.right.label]
            
            
            
    def simplify_grad(self, matrices_and_functions, grad):
        # Matrices cannot be similified obviously
        if isinstance(grad, Matrix):
            return
        
        # If the function is a transpose, we may be able to simplify it
        if isinstance(grad, Transpose):
            self.simplify_grad(matrices_and_functions, grad.matrix)
            self.simplify_transpose(matrices_and_functions, grad)
            
        # If the function is a Matmul
        if isinstance(grad, Matmul):
            # Simplify left and right, then simplify the matmul
            self.simplify_grad(matrices_and_functions, grad.left)
            self.simplify_grad(matrices_and_functions, grad.right)
            self.simplify_matmul(matrices_and_functions, grad)

    
    def simplify(self, matrices_and_functions):
        # Get all gradient functions
        functions = {i:j.grad for i, j in matrices_and_functions.items() if isinstance(j, Matrix)}
        
        # Iterate through all the function gradients
        for matrix, function in functions.items():
            # Iterate through all the gradient functions
            for grad in function:
                self.simply_grad(matrices_and_functions, grad)