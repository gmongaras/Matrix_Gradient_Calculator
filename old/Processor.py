


class Processor:
    def __init__(self):
        pass

    def process(self, parsed_input):
        # Intialize the current shape to the first matrix
        cur_shape = parsed_input[0].shape
        prev_op = parsed_input[0]
        
        # Iterate through the parsed input
        for matrix in parsed_input[1:]:
            if matrix.hadamard_on_left:
                assert len(cur_shape) == len(matrix.shape), f"Matrix shapes for hadamard should match. Error at: {prev_op} and {matrix}."
                for s in range(0, len(cur_shape)):
                    assert cur_shape[s] == matrix.shape[s], f"Matrix shapes for hadamard should match. Error at: {prev_op} and {matrix}."
                # Shape stays the same
            else:
                # Multiply matrices on the right to get a new shape
                assert cur_shape[-1] == matrix.shape[-2], f"Matrix shapes are not compatible for multiplication. Error at: {prev_op} and {matrix}."
                cur_shape = cur_shape[:-1] + [matrix.shape[-1]]
                prev_op = matrix
        
        return cur_shape