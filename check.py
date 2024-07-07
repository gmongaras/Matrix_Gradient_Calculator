import torch



class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        ctx.save_for_backward(Q, K, V)
        # return Q @ K.mT @ V
        return K.mT @ Q @ K.mT @ V

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V = ctx.saved_tensors
        Q_grad = K @ grad_output @ V.mT @ K
        K_grad = V @ grad_output.mT @ K.mT @ Q + Q @ K.mT @ V @ grad_output.mT
        V_grad = K @ Q.mT @ K @ grad_output
        # Q_grad = grad_output @ V.mT @ K
        # K_grad = V @ grad_output.mT @ Q
        # V_grad = K @ Q.mT @ grad_output
        return Q_grad, K_grad, V_grad
    
    

N, H, S, D = 1, 2, 16, 12
Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
K = torch.rand(N, H, S, D, requires_grad=True).cuda()
V = torch.rand(N, H, S, D, requires_grad=True).cuda()
torch.autograd.gradcheck(Function.apply, (Q.double(), K.double(), V.double()), eps=1e-4)