from Matrix import Matrix
from copy import deepcopy


class Calculator:
    def __init__(self,):
        pass
    
    
    # Transposes an entire sequence of matrices
    # This reverses the order of the matrices and transposes each one
    def transpose_sequence(self, sequence):
        transposed = []
        for matrix in sequence[::-1]:
            transposed.append(matrix.transpose_())
        return transposed
    
    
    def grad_wrt_left(self, left, right, prev_grad):
        return prev_grad + self.transpose_sequence(right)
    
    def grad_wrt_right(self, left, right, prev_grad):
        return self.transpose_sequence(left) + prev_grad
    
    
    def hadamard_grad_wrt_left(self, left, right, prev_grad):
        # Derivative wrt to left of hadamard results in a hadamard with the gradient and the right
        right_ = deepcopy(right)
        right_[0].hadamard_on_left = True
        return prev_grad + right
    
    def hadamard_grad_wrt_right(self, left, right, prev_grad):
        # Derivative wrt to right of hadamard results in a hadamard with the left and the gradient
        prev_grad_ = deepcopy(prev_grad)
        prev_grad_[0].hadamard_on_left = True
        return left + prev_grad_
    
    
    def combine_matrices(self, matrices):
        output_string = ""
        for matrix in matrices[:-1]:
            output_string += f"{matrix} "
        output_string += str(matrices[-1])
        return output_string
    
    def calculate(self, parsed_input, final_shape):
        # Previous gradient
        prev_grad = Matrix("dL", final_shape)
        
        gradients = {}
        
        cur_grad = [prev_grad]
        
        # Iterate from right to left, thus we always multiply on the right
        # and use the right multiplication rule.
        for i in range(len(parsed_input) - 1, -1, -1):
            symbol = parsed_input[i]
            
            # Normal matrix
            if symbol.power is None:
                # Left is going to be the sequence up to the current matrix
                left = parsed_input[:i]
                # Right is just the current matrix
                right = [symbol]
                
                # is this operation a hadamard product?
                if len(left) > 0 and right[0].hadamard_on_left == True:
                    # Matrix gradient wrt right is the gradient for the right matrix
                    mat_grad = self.hadamard_grad_wrt_right(left, right, cur_grad)
                    
                    # Matrix gradient wrt left is the chain rule gradient
                    if i > 0:
                        cur_grad = self.hadamard_grad_wrt_left(left, right, cur_grad)
                
                # Normal matrix multiplication
                else:
                    # Matrix gradient wrt right is the gradient for the right matrix
                    mat_grad = self.grad_wrt_right(left, right, cur_grad)
                    
                    # Matrix gradient wrt left is the chain rule gradient
                    if i > 0:
                        # if left[0].is_transposed:
                        #     raise Exception("This is not implemented yet.")
                        cur_grad = self.grad_wrt_left(left, right, cur_grad)
                        
                # If transposed, we need to transpose the entire output gradient
                if right[0].is_transposed:
                    mat_grad = self.transpose_sequence(mat_grad)
                try:
                    gradients[right[0].label].append(mat_grad)
                except KeyError:
                    gradients[right[0].label] = [mat_grad]
                
        return gradients