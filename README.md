# Matrix_Gradient_Calculator
I feel like building a calculator for matrix gradients, but where the function has an ambiguous input gradient like in a machine learning backprop context.


Ever find yourself needing the calculate the gradient of a function for PyTorch and you funciton lloks like this garbage?

(M^10 Q K^T M^10) (V * W)

I have and that's why I'm making this. So that I don't have to again.

Current partial support:
- matrix multiplication
- hadamard product
- transposes

Less than partial support:
- matrix power

Future support:
- sums