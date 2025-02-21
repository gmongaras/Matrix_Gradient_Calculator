from classes.Matrix import Matrix
from classes.Operations import Transpose, Power, Matmul, Hadamard, Add, MatrixFunction, MatrixVectorFunction
from Parser import special_letters
from helpers import get_matching_bracket





class Processor:
    def __init__(self,):
        # Name of function will start from f0 and monotonically increase
        self.function_number = 0
        
        self.special_letters = special_letters
    
    
    
    def add_function(self, function, matrices_and_functions):
        matrices_and_functions[f"f{self.function_number}"] = function
        self.function_number += 1
        
        
    def get_function_name(self):
        return f"f{self.function_number}"
        
        
        
    def process_parenthesis(self, sequence, shapes, function_types, matrices_and_functions):
        # Get the closing parenthesis
        closing_parenthesis = get_matching_bracket(sequence, 0)
        
        # Is there a transpose after the closing parenthesis?
        if closing_parenthesis + 1 < len(sequence) and sequence[closing_parenthesis + 1] == "^" and closing_parenthesis + 2 < len(sequence) and sequence[closing_parenthesis + 2] == "T":
            # Slice string for recursion
            substring = sequence[:closing_parenthesis].strip()
            # Process the inner string
            tmp_right = self.process(substring, shapes, function_types, matrices_and_functions=matrices_and_functions)[0]
            # Transpose the matrix
            tmp_right = Transpose(tmp_right, name=self.get_function_name)
            self.add_function(tmp_right, matrices_and_functions)
            return tmp_right, closing_parenthesis + 3
        else:
            # Slice string for recursion
            substring = sequence[:closing_parenthesis].strip()
            # Process the inner string
            return self.process(substring, shapes, function_types, matrices_and_functions=matrices_and_functions)[0], closing_parenthesis + 1
    
    
    
    def process_matrix_function(self, sequence, shapes, function_types, matrices_and_functions):
        # We need to get the entire function name
        i = 0
        function_name = ""
        # Iterate until we reach a left parenthesis
        while i < len(sequence) and sequence[i] != "(":
            symbol = sequence[i]
            
            # If the symbol is not a letter, raise an error
            if not symbol.isalpha() and symbol not in self.special_letters:
                raise ValueError("Function name must be all letters")
            
            function_name += symbol
            i += 1
        i += 1
        
        # Get the closing parenthesis
        closing_parenthesis = get_matching_bracket(sequence, i)
        # Slice the string for recursion
        substring = sequence[i:closing_parenthesis]
        # Process the function
        inner_function = self.process(substring, shapes, function_types, matrices_and_functions=matrices_and_functions)[0]
        
        # Wrap the function in a MatrixFunction or MatrixVectorFunction depending on the function type
        if function_types[function_name] == "vector":
            current_right = MatrixVectorFunction(inner_function, name=function_name)
        elif function_types[function_name] == "scalar":
            current_right = MatrixFunction(inner_function, name=function_name)
        else:
            raise ValueError(f"Unknown function type: {function_types[function_name]}")
        
        return current_right, closing_parenthesis
    
    
    
    def process_matrix(self, symbol, sequence, shapes, function_types, matrices_and_functions):
        # If we have not encountered this matrix yet, add it
        if symbol not in matrices_and_functions:
            matrices_and_functions[symbol] = Matrix(shapes[symbol], symbol)
            
        current_right = matrices_and_functions[symbol]
        
        # Is there a "^T" after? If so, transpose this matrix
        i = 0
        if i < len(sequence) and sequence[i] == "^":
            # Is it transposed?
            if i + 1 < len(sequence) and sequence[i + 1] == "T":
                current_right = Transpose(matrices_and_functions[symbol], name=self.get_function_name)
                self.add_function(current_right, matrices_and_functions)
                i += 2
            # Is it a power?
            elif i + 1 < len(sequence) and sequence[i + 1].isnumeric():
                # We need to get the entire power
                num = ""
                while i + 1 < len(sequence) and sequence[i + 1].isnumeric():
                    num += sequence[i + 1]
                    i += 1
                i += 1
                power = int(num)
                current_right = Power(matrices_and_functions[symbol], power, name=self.get_function_name)
                
        return current_right, i
    
    
    
    def process(self, sequence, shapes, function_types, matrices_and_functions=None):
        if matrices_and_functions == None:
            matrices_and_functions = {}
            
            
        i = 1
        op = Matmul
        
        # Initialize with the first
        current_left = None
        # matrices_and_functions[sequence[0]] = Matrix(sequence[0], shapes[sequence[0]])
        if sequence[0].isalpha() and sequence[0].isupper():
            current_right, i_add = self.process_matrix(sequence[0], sequence[1:], shapes, function_types, matrices_and_functions)
        elif sequence[0].isalpha() and sequence[0].islower():
            current_right, i_add = self.process_matrix_function(sequence, shapes, function_types, matrices_and_functions)
        elif sequence[0] == "(":
            current_right, i_add = self.process_parenthesis(sequence[1:], shapes, function_types, matrices_and_functions)
        i += i_add
        
        # Iterate from left to right which is going from the internal fnction to the outer
        while i < len(sequence):
            # Get next symbol
            symbol = sequence[i]
            i += 1
            
            # Skip spaces
            if symbol == " ":
                continue
            
            # If we see "@", then the operation becomes Matmul
            elif symbol == "@":
                op = Matmul
                continue
            
            # If we see "*", then the operation becomes Hadamard
            elif symbol == "*":
                op = Hadamard
                continue
            
            # If we see "+", then the operation becomes Add
            elif symbol == "+":
                op = Add
                continue
            
            # If we see a parenthesis, we can just recursively call this function
            elif symbol == "(":
                # Move right to current left
                current_left = current_right
                
                current_right, i_add = self.process_parenthesis(sequence[i:], shapes, function_types, matrices_and_functions)
                i += i_add
                
            
            # Is the symbol an uppercase letter? If so, then it's a matrix
            elif symbol.isalpha() and symbol.isupper():
                # Move right to current left
                current_left = current_right
                
                current_right, i_add = self.process_matrix(symbol, sequence[i:], shapes, function_types, matrices_and_functions)
                i += i_add
                
                
            # Is this the start of a function?
            elif symbol.isalpha() and symbol.islower():
                # Process the function
                current_right, i_add = self.process_matrix_function(sequence[i:], shapes, function_types, matrices_and_functions)
                i += i_add
                        
                        
            # Create a matrix multiplication operation with the current left and the right matrix
            current_right = op(current_left, current_right, name=self.get_function_name)
            self.add_function(current_right, matrices_and_functions)
            op = Matmul
                
                
        return current_right, matrices_and_functions