import torch



class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W):
        ctx.save_for_backward(Q, K, V, W)
        # return Q @ K.mT @ V
        # return K.mT @ Q @ K.mT @ V
        # return V * V * W
        return K.mT @ Q @ K.mT @ V * W

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W = ctx.saved_tensors
        # Q_grad = K @ grad_output @ V.mT @ K
        # K_grad = V @ grad_output.mT @ K.mT @ Q + Q @ K.mT @ V @ grad_output.mT
        # V_grad = K @ Q.mT @ K @ grad_output
        
        # Q_grad = grad_output @ V.mT @ K
        # K_grad = V @ grad_output.mT @ Q
        # V_grad = K @ Q.mT @ grad_output
        
        W_grad = K.mT @ Q @ K.mT @ V * grad_output
        V_grad = K @ Q.mT @ K @ grad_output * W
        K_grad = V * W.mT @ grad_output.mT @ K.mT @ Q + Q @ K.mT @ V * W.mT @ grad_output.mT
        Q_grad = K @ grad_output * W @ V.mT @ K
                
        return Q_grad, K_grad, V_grad, W_grad
    
    

N, H, S, D = 1, 2, 16, 12
Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
K = torch.rand(N, H, S, D, requires_grad=True).cuda()
V = torch.rand(N, H, S, D, requires_grad=True).cuda()
W = torch.rand(N, H, D, D, requires_grad=True).cuda()
torch.autograd.gradcheck(Function.apply, (Q.double(), K.double(), V.double(), W.double()), eps=1e-4)