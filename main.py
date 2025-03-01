from classes.Matrix import Matrix
from classes.Operations import Transpose, Matmul
from Parser import Parser
from Processor import Processor
from Calculator import Calculator
from Similifier import Similifier






# sequence = "(M^10 Q K^T M^10) (V * W)"
# sequence = "softmax((Q K^T) * M) V"
# sequence = "silu((Q K^T) * M) V"
# shapes = {
#     "Q": ["N", "S", "d_k"],
#     "K": ["N", "S", "d_k"],
#     "V": ["N", "S", "d_v"],
#     "M": ["N", "S", "S"],
#     "W": ["N", "d_k", "d_k"],
# }
# wanted_grads = [
#     "Q",
#     "K",
#     "V",
#     "M",
#     "W",
# ]
# function_types = {
#     "softmax": "vector",
#     "silu": "scalar",
# }




sequence = "softmax(((X Q) (X K)^T) * M) (X V)"
shapes = {
    "X": ["N", "S", "d"],
    "Q": ["d", "d_k"],
    "K": ["d", "d_k"],
    "V": ["d", "d_v"],
    "M": ["N", "S", "S"],
}
wanted_grads = [
    "X",
    "Q",
    "K",
    "V",
    "M",
]
function_types = {
    "softmax": "vector",
    "silu": "scalar",
}






# sequence = "silu((Q K^T) * M) V"
# shapes = {
#     "Q": ["N", "S", "d_k"],
#     "K": ["N", "S", "d_k"],
#     "V": ["N", "S", "d_v"],
#     "M": ["N", "S", "S"],
#     "W": ["N", "S", "d_v"],
# }
# wanted_grads = [
#     "Q",
#     "K",
#     "V",
#     "M",
#     # "W",
# ]

# sequence = "f(f(f((f(X @ A) + X) @ B) @ C) @ D)"
# shapes = {
#     "X": ["1", "d"],
#     "A": ["d", "d"],
#     "B": ["d", "D"],
#     "C": ["D", "D"],
#     "D": ["D", "d"],
# }
# wanted_grads = [
#     "X",
#     "A",
#     "B",
#     "C",
#     "D",
# ]



def main(sequence, shapes, wanted_grads, function_types):
    parser = Parser()
    processor = Processor()
    calculator = Calculator()
    similifier = Similifier()


    # Parse the input for errors
    grad_shape = parser.parse(sequence, shapes, function_types)

    print(f"Input gradient dL will be of shape {grad_shape}")

    # Process input to a symbolic representation
    symbols, matrices_and_functions = processor.process(sequence, shapes, function_types)

    # Prev grad matrix initialized to the output shape
    prev_grad = Matrix(grad_shape, "dL")

    # Calculate the gradients
    calculator.calculate(symbols, matrices_and_functions, prev_grad)

    # Clone gradients so there's no dependencies between them
    for key in wanted_grads:
        matrices_and_functions[key].grad = [matrix.copy() for matrix in matrices_and_functions[key].grad]

    # Simplify the gradients
    similifier.simplify(matrices_and_functions)
    
    return matrices_and_functions

matrices_and_functions = main(
    sequence,
    shapes,
    wanted_grads,
    function_types
)

# Print the gradients
for key in wanted_grads:
    grad_fns = matrices_and_functions[key].grad
    string = "+".join([str(grad_fn) for grad_fn in grad_fns])
    print(f"Gradient of {key}: {string}")
print()
# Print the gradients for torch
for key in wanted_grads:
    grad_fns = matrices_and_functions[key].grad
    string = "+".join([grad_fn.torch_str() for grad_fn in grad_fns]).replace("dL", "prev_grad")
    print(f"{key}_grad = {string}")
    
# # Print gradients in torch notation
# for key in wanted_grads:
#     grad_fns = matrices_and_functions[key].grad
#     string = " + ".join([str(grad_fn) for grad_fn in grad_fns])
#     string = string.replace(" ", "@")
#     string = string.replace("@*@", " * ")
#     string = string.replace("@+@", " + ")
#     string = string.replace("@", " @ ")
#     string = string.replace("[", "(")
#     string = string.replace("]", ")")
#     string = string.replace("dL", "prev_grad")
#     string = string.replace("^T", ".mT")
#     string = string.replace("'", "_der")
#     print(f"{key}_grad = {string}")