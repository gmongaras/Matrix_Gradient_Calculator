from Matrix import Matrix
from copy import deepcopy

class Parser():
    def __init__(self):
        self.special = ["^", "T", "*"]
    
    def parse(self, equation, shapes):
        # Sequence of operations
        operations = []
        
        # Previous cache for the parser if needed
        cache = []
        
        # Iterate over every character in the input string
        c = 0
        hadamard_op = False
        while c < len(equation):
            char = equation[c]
            c += 1
            
            # Skip spaces
            if char == " ":
                continue
            
            # Is the character a non special letter?
            elif char.isalpha() and char not in self.special:
                print("Letter: " + char)
                
                operations.append(Matrix(char, deepcopy(shapes[char]), hadamard_on_left=hadamard_op))
                cache = [operations[-1]]
                hadamard_op = False
                
            # Is the character a T?
            elif char == "T":
                print("T")
                
            # Is the character an up arrow?
            elif char == "^":
                
                # We expect the next character to to either be a number or a T
                if c < len(equation):
                    next_char = equation[c]
                    c += 1
                    
                    if next_char.isdigit():
                        # Iterate until we find the end of the number
                        for i in range(c, len(equation)):
                            if equation[i].isdigit():
                                c += 1
                                next_char += equation[i]
                            else:
                                break
                        
                        # This is a power
                        cache[0].power = int(next_char)
                    
                    # This is a transposition
                    elif next_char == "T":
                        # Mark the matrix as transposed
                        cache[0].transpose()
                    else:
                        raise Exception("Invalid character after power at index " + str(c))
                    
        
            elif char == "*":
                # Change the flag indicating there is a hadamard on the left
                hadamard_op = True


            # Is the character a special character?
            elif char in self.special:
                raise Exception("Invalid character at index " + str(c) + "is a special character.")


            else:
                raise Exception("Invalid character at index " + str(c))
            
            
        return operations