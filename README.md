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




Was it worth it? We now know the gradient of softmax attention

```
N, H, S, D = 1, 2, 16, 12
Q = torch.rand(D, D, requires_grad=True).cuda()
K = torch.rand(D, D, requires_grad=True).cuda()
V = torch.rand(D, D, requires_grad=True).cuda()
X = torch.rand(N, H, S, D, requires_grad=True).cuda()
M = torch.rand(N, H, S, S, requires_grad=True).cuda()

torch.nn.functional.softmax(((X @ Q) @ (X @ K).mT) * M, dim=-1) @ (X @ V)

softmax = torch.nn.functional.softmax
Jacobian = lambda x: torch.diag_embed(x) - x[..., :, None] * x[..., None, :]

X_grad = ((torch.einsum("...ijk,...ik->...ij", Jacobian(softmax(((X @ Q @ (X @ K).mT) * M), dim=-1)), prev_grad @ V.mT @ X.mT)) * M) @ X @ K @ Q.mT+((torch.einsum("...ijk,...ki->...ji", Jacobian(softmax(((X @ Q @ (X @ K).mT) * M), dim=-1)), X @ V @ prev_grad.mT)) * (M.mT)) @ X @ Q @ K.mT+softmax(((X @ K @ Q.mT @ X.mT) * (M.mT)), dim=-2) @ prev_grad @ V.mT
Q_grad = X.mT @ ((torch.einsum("...ijk,...ik->...ij", Jacobian(softmax(((X @ Q @ (X @ K).mT) * M), dim=-1)), prev_grad @ V.mT @ X.mT)) * M) @ X @ K
K_grad = X.mT @ ((torch.einsum("...ijk,...ki->...ji", Jacobian(softmax(((X @ Q @ (X @ K).mT) * M), dim=-1)), X @ V @ prev_grad.mT)) * (M.mT)) @ X @ Q
V_grad = X.mT @ softmax(((X @ K @ Q.mT @ X.mT) * (M.mT)), dim=-2) @ prev_grad
M_grad = ((X @ Q @ K.mT @ X.mT) * (torch.einsum("...ijk,...ik->...ij", Jacobian(softmax(((X @ Q @ (X @ K).mT) * M), dim=-1)), prev_grad @ V.mT @ X.mT)))
```

```
N, H, S, D = 1, 2, 16, 12
Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
K = torch.rand(N, H, S, D, requires_grad=True).cuda()
V = torch.rand(N, H, S, D, requires_grad=True).cuda()
M = torch.rand(N, H, S, S, requires_grad=True).cuda()

torch.nn.functional.softmax(((X @ Q) @ (X @ K).mT) * M, dim=-1) @ (X @ V)

softmax = torch.nn.functional.softmax
Jacobian = lambda x: torch.diag_embed(x) - x[..., :, None] * x[..., None, :]

Q_grad = ((torch.einsum("...ijk,...ik->...ij", Jacobian(softmax(((Q @ K.mT) * M), dim=-1)), prev_grad @ V.mT)) * M) @ K
K_grad = ((torch.einsum("...ijk,...ki->...ji", Jacobian(softmax(((Q @ K.mT) * M), dim=-1)), V @ prev_grad.mT)) * (M.mT)) @ Q
V_grad = softmax(((K @ Q.mT) * (M.mT)), dim=-2) @ prev_grad
M_grad = ((Q @ K.mT) * (torch.einsum("...ijk,...ik->...ij", Jacobian(softmax(((Q @ K.mT) * M), dim=-1)), prev_grad @ V.mT)))
```


These are actually correct. Just look absolutely terrible.


I need to clean this messy repo up.




Current partial support:
- matrix multiplication
- hadamard product
- additions
- transposes
- scalar functions
- vector functions
- torch compilation instead of just replacing strings

Less than partial support:
- matrix power (I don't remember why this is here. Hopefully it doesn't become a problem)

Future support:
- matrix names
- better errors
- actual testing