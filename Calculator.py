from classes.Matrix import Matrix
from classes.Operations import Transpose, Power, Matmul, Hadamard





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
            if isinstance(symbols.left, Power):
                left_grad = symbols.left.get_grad_wrt_left(prev_grad, other_side=symbols.right, operation="matmul" if isinstance(symbols, Matmul) else "hadamard")
                self.calculate(symbols.left.matrix, matrices_and_functions, left_grad)
            else:
                left_grad = symbols.get_grad_wrt_left(prev_grad)
                self.calculate(symbols.left, matrices_and_functions, left_grad)
                
            if isinstance(symbols.right, Power):
                right_grad = symbols.right.get_grad_wrt_right(prev_grad, other_side=symbols.left, operation="matmul" if isinstance(symbols, Matmul) else "hadamard")
                self.calculate(symbols.right.matrix, matrices_and_functions, right_grad)
            else:
                right_grad = symbols.get_grad_wrt_right(prev_grad)
                self.calculate(symbols.right, matrices_and_functions, right_grad)
            
                
        # If the function is a matrix, we add the gradient to the matrix
        elif isinstance(symbols, Matrix):
            symbols.add_grad(Transpose(prev_grad) if is_transposed else prev_grad)