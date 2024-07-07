


class Processor:
    def __init__(self):
        pass

    def process(self, parsed_input):
        # Intialize the current shape to the first matrix
        cur_shape = parsed_input[0].shape
        prev_op = parsed_input[0]
        
        # Iterate through the parsed input
        for operation in parsed_input[1:]:
            # Multiply matrices on the right to get a new shape
            assert cur_shape[-1] == operation.shape[-2], f"Matrix shapes are not compatible for multiplication. Error at: {prev_op} and {operation}."
            cur_shape = cur_shape[:-1] + [operation.shape[-1]]
            prev_op = operation
        
        return cur_shape