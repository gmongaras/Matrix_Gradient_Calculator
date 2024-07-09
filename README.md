# Matrix_Gradient_Calculator
I feel like building a calculator for matrix gradients, but where the function has an ambiguous input gradient like in a machine learning backprop context.


Ever find yourself needing the calculate the gradient of a function for PyTorch and you funciton lloks like this garbage?

(M^10 Q K^T M^10) (V * W)

I have and that's why I'm making this. So that I don't have to again.




I guess I can now get all the gradients for a silu neural network:

```
O = f(f(f((f(X @ A) + X) @ B) @ C) @ D)

shapes = {
    "X": ["1", "d"],
    "A": ["d", "d"],
    "B": ["d", "D"],
    "C": ["D", "D"],
    "D": ["D", "d"],
}

X_grad = ((f_der(X @ A)) * (((f_der(((f(X @ A)) + X) @ B)) * (((f_der(f(((f(X @ A)) + X) @ B) @ C)) * (((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad) @ D.mT)) @ C.mT)) @ B.mT)) @ A.mT + ((f_der(((f(X @ A)) + X) @ B)) * (((f_der(f(((f(X @ A)) + X) @ B) @ C)) * (((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad) @ D.mT)) @ C.mT)) @ B.mT
A_grad = X.mT @ ((f_der(X @ A)) * (((f_der(((f(X @ A)) + X) @ B)) * (((f_der(f(((f(X @ A)) + X) @ B) @ C)) * (((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad) @ D.mT)) @ C.mT)) @ B.mT))
B_grad = (((f(X @ A)) + X)).mT @ ((f_der(((f(X @ A)) + X) @ B)) * (((f_der(f(((f(X @ A)) + X) @ B) @ C)) * (((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad) @ D.mT)) @ C.mT))
C_grad = (f(((f(X @ A)) + X) @ B)).mT @ ((f_der(f(((f(X @ A)) + X) @ B) @ C)) * (((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad) @ D.mT))        
D_grad = (f(f(((f(X @ A)) + X) @ B) @ C)).mT @ ((f_der(f(f(((f(X @ A)) + X) @ B) @ C) @ D)) * prev_grad)
```




Current partial support:
- matrix multiplication
- hadamard product
- additions
- transposes
- scalar functions

Less than partial support:
- matrix power (I don't remember why this is here. Hopefully it doesn't become a problem)

Future support:
- matrix names
- vector functions