from classes.Matrix import Matrix
from classes.Operations import Transpose, Matmul, Hadamard





class Calculator:
    def __init__(self, ):
        pass
    
    
    def calculate(self, symbols, matrices_and_functions, prev_grad, is_transposed = False):
        # We just calculate the gradients from inside to outside
        
        # If the function is a transpose, we just calculate the gradient and use this
        # as our new previous gradient
        if isinstance(symbols, Transpose):
            prev_grad = prev_grad
            self.calculate(symbols.matrix, matrices_and_functions, prev_grad, is_transposed=not is_transposed)
            
        # If the function is Matmul, Hadamard, or Add, we calculate both sides and
        # chain rule if either side is a function
        elif isinstance(symbols, Matmul) or isinstance(symbols, Hadamard):
            left_grad = symbols.get_grad_wrt_left(prev_grad)
            right_grad = symbols.get_grad_wrt_right(prev_grad)
            
            # If left is a function, calculate its gradient
            self.calculate(symbols.left, matrices_and_functions, left_grad)
                
            # If right is a function, calculate its gradient
            self.calculate(symbols.right, matrices_and_functions, right_grad)
                
        # If the function is a matrix, we add the gradient to the matrix
        elif isinstance(symbols, Matrix):
            symbols.add_grad(Transpose(prev_grad) if is_transposed else prev_grad)