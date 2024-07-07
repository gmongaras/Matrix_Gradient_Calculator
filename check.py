import torch



class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, M):
        ctx.save_for_backward(Q, K, V, W, M)
        return ((Q @ K.mT) * M) @ V

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W, M = ctx.saved_tensors
        Q_grad = ((grad_output @ V.mT) * M) @ K.mT.mT
        K_grad = (Q.mT @ ((grad_output @ V.mT) * M)).mT
        V_grad = ((Q @ K.mT) * M).mT @ grad_output
        W_grad = None
        M_grad = ((Q @ K.mT) * (grad_output @ V.mT))
        return Q_grad, K_grad, V_grad, W_grad, M_grad
    
    

N, H, S, D = 1, 2, 16, 12
Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
K = torch.rand(N, H, S, D, requires_grad=True).cuda()
V = torch.rand(N, H, S, D, requires_grad=True).cuda()
W = torch.rand(N, H, S, D, requires_grad=True).cuda()
M = torch.rand(N, H, S, S, requires_grad=True).cuda()
torch.autograd.gradcheck(Function.apply, (Q.double(), K.double(), V.double(), W.double(), M.double()), eps=1e-4)