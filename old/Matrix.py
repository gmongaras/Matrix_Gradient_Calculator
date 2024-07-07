from copy import deepcopy




class Matrix:
    def __init__(self, label, shape, is_transposed=False, power=None, hadamard_on_left=False):
        self.label = label
        self.shape = shape
        self.is_transposed = is_transposed
        self.power = power
        self.hadamard_on_left = hadamard_on_left
        
        
    def transpose(self):
        self.is_transposed = not self.is_transposed
        
        # Transpose last two dims
        tmp = self.shape[-1]
        self.shape[-1] = self.shape[-2]
        self.shape[-2] = tmp
        
        
    def transpose_(self):
        # Create new instance of the matrix
        new_matrix = deepcopy(self)
        
        # Transpose the matrix
        new_matrix.transpose()
        
        return new_matrix
        
        
    def add_power(self, power):
        if power != 1:
            self.power = power
            
            
    def __str__(self):
        output_string = self.label
        if self.is_transposed:
            output_string += "^T"
        if self.power:
            output_string += f"^{self.power}"
        if self.hadamard_on_left:
            output_string = f"* {output_string}"
        return output_string