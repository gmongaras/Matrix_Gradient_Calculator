from copy import deepcopy
from classes.Operations import Transpose, Matmul, Hadamard, Add




operations = {
    "@": "matmul",
    "*": "hadamard",
    "+": "add",
}


op_to_function = {
    "matmul": Matmul,
    "hadamard": Hadamard,
    "transpose": Transpose,
    "add": Add,
}

# Special letters for matrix funcitons that are allowed
special_letters = ["_", "-", ".", "!", "<", ">"]



class Parser:
    def __init__(self):
        pass
    
    
    def parse(self, sequence, shapes):
        # Remove all spaces
        sequence = sequence.replace(" ", "")
        
        # Operation assumed to be matmul
        cur_op = "matmul"
        
        # Current shape will be initialized as None
        current_shape = None
        
        # First check that there are the same number of opening and closing parenthesis
        assert sequence.count("(") == sequence.count(")"), "Number of opening and closing parenthesis do not match"
        
        i = 0
        while i < len(sequence):
            # Get the next symbol
            symbol = sequence[i]
            i += 1
            
            # Is this a matrix? A matrix can only be a capital letter
            if symbol.isalpha() and symbol.isupper():
                # Get the shape of the matrix
                shape = deepcopy(shapes[symbol])
                
                # Is the next symbol an up arrow?
                if i < len(sequence) and sequence[i] == "^":
                    # Is the symbol after the up arrow a T?
                    if i + 1 < len(sequence) and sequence[i + 1] == "T":
                        # If so, we have a transpose
                        i += 2
                        
                        # Simulate the transpose on the matrix shape
                        shape = op_to_function["transpose"].simulate(shape)
                        
                    # Is the symbol after the up arrow a number?
                    elif i + 1 < len(sequence) and sequence[i + 1].isnumeric():
                        i += 1
                        # Iterate through the number
                        num = ""
                        while i < len(sequence) and sequence[i].isnumeric():
                            num += sequence[i]
                            i += 1
                        # If so, we have a power
                    else:
                        raise ValueError("Invalid syntax: ^ must be followed by T or a number")
                
                # Simulate the operation
                if current_shape is None:
                    current_shape = shape
                else:
                    current_shape = op_to_function[cur_op].simulate(current_shape, shape)
                # Reset to matmul
                cur_op = "matmul"
                
            # Is this the start of a function?
            elif symbol.isalpha() and symbol.islower():
                # We need to get the entire function name
                function_name = symbol
                # Iterate until we reach a left parenthesis
                while i < len(sequence) and sequence[i] != "(":
                    symbol = sequence[i]
                    
                    # If the symbol is not a letter, raise an error
                    if not symbol.isalpha() and symbol not in special_letters:
                        raise ValueError("Function name must be all letters")
                    
                    function_name += symbol
                    i += 1
                i += 1
                
                # Get the closing parenthesis (last instance)
                closing_parenthesis = len(sequence[i:]) - 1 - sequence[i:][::-1].index(")")
                # Slice the string for recursion
                substring = sequence[i:i+closing_parenthesis]
                # Process the function
                current_shape = self.parse(substring, shapes)
                # Skip the closing parenthesis
                i = i + closing_parenthesis + 1
                    
            # Is this an operation
            elif symbol in operations:
                cur_op = operations[symbol]
                
            # Is this an opening parenthesis?
            elif symbol == "(":
                # Get the closing parenthesis (last instance)
                closing_parenthesis = len(sequence[i:]) - 1 - sequence[i:][::-1].index(")")
                # Slice the string for recursion
                substring = sequence[i:i+closing_parenthesis]
                # Process the parenthesis
                output_shape = self.parse(substring, shapes)
                # Skip the closing parenthesis
                i = i + closing_parenthesis + 1
                
                # Simulate the operation with the output shape
                if current_shape is None:
                    current_shape = output_shape
                else:
                    current_shape = op_to_function[cur_op].simulate(current_shape, output_shape)
                    
            else:
                raise ValueError(f"Unknown symbol: {symbol} at index {i}")
            
        return current_shape