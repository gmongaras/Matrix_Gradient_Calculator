from classes.Matrix import Matrix
from classes.Operations import Transpose, Matmul
from Parser import Parser
from Processor import Processor
from Calculator import Calculator






sequence = "(Q K^T * M) (V * W)"
shapes = {
    "Q": ["N", "S", "d_k"],
    "K": ["N", "S", "d_k"],
    "V": ["N", "S", "d_v"],
    "M": ["N", "S", "S"],
    "W": ["N", "S", "d_v"],
}
wanted_grads = [
    "Q",
    "K",
    "V",
    "M",
    "W",
]




parser = Parser()
processor = Processor()
calculator = Calculator()


# Parse the input for errors
grad_shape = parser.parse(sequence, shapes)

print(f"Input gradient dL will be of shape {grad_shape}")

# Process input to a symbolic representation
symbols, matrices_and_functions = processor.process(sequence, shapes)

# Prev grad matrix initialized to the output shape
prev_grad = Matrix(grad_shape, "dL")

# Calculate the gradients
calculator.calculate(symbols, matrices_and_functions, prev_grad)

# Print the gradients
for key in wanted_grads:
    grad_fns = matrices_and_functions[key].grad
    string = "+".join([str(grad_fn) for grad_fn in grad_fns])
    print(f"Gradient of {key}: {string}")
    
# Print gradients in torch notation
for key in wanted_grads:
    grad_fns = matrices_and_functions[key].grad
    string = " + ".join([str(grad_fn) for grad_fn in grad_fns])
    string = string.replace(" ", "@")
    string = string.replace("@*@", " * ")
    string = string.replace("@+@", " + ")
    string = string.replace("@", " @ ")
    string = string.replace("[", "(")
    string = string.replace("]", ")")
    string = string.replace("dL", "prev_grad")
    string = string.replace("^T", ".mT")
    print(f"{key}_grad = {string}")