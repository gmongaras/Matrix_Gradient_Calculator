from Parser import Parser
from Processor import Processor
from Calculator import Calculator



parser = Parser()
processor = Processor()
calculator = Calculator()



equation = "K^T Q K^T V * W"
shapes = {
    "Q": ["N", "S", "d_k"],
    "K": ["N", "S", "d_k"],
    "V": ["N", "S", "d_v"],
    "W": ["N", "d_k", "d_v"]
}


# Parse the input
parsed_input = parser.parse(equation, shapes)

# Process the input and check for errors
final_shape = processor.process(parsed_input)

# Print the final shape
print(f"Input gradient dL will be of shape {final_shape}")

# Calculate the gradients for each matrix
gradients = calculator.calculate(parsed_input, final_shape)

# Iterate through all the gradients and convert them to strings
for key, value in gradients.items():
    total_grad = []
    for grad in value:
        total_grad.append(calculator.combine_matrices(grad))
    total_grad = " + ".join(total_grad)
    print(f"dL/d{key} = {total_grad}")
