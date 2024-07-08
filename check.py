import torch



class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, M):
        ctx.save_for_backward(Q, K, V, W, M)
        # return ((Q @ K.mT) * M) @ V
        # return (Q @ K.mT * M) @ (V * W)
        # return (Q @ K.mT @ torch.matrix_power(M, 10)) @ (V * W)
        return (torch.matrix_power(M, 10) @ Q @ K.mT @ torch.matrix_power(M, 10)) @ (V * W)

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W, M = ctx.saved_tensors
        # Q_grad = ((grad_output @ V.mT) * M) @ K.mT.mT
        # K_grad = (Q.mT @ ((grad_output @ V.mT) * M)).mT
        # V_grad = ((Q @ K.mT) * M).mT @ grad_output
        # W_grad = None
        # M_grad = ((Q @ K.mT) * (grad_output @ V.mT))
        
        prev_grad = grad_output
        
        # Q_grad = ((prev_grad @ ((V * W)).mT) * M) @ (K.mT).mT
        # K_grad = (Q.mT @ ((prev_grad @ ((V * W)).mT) * M)).mT
        # V_grad = (((((Q @ K.mT) * M)).mT @ prev_grad) * W)
        # M_grad = ((Q @ K.mT) * (prev_grad @ ((V * W)).mT))
        # W_grad = (V * ((((Q @ K.mT) * M)).mT @ prev_grad))
        
        
        # Q_grad = prev_grad @ ((V * W)).mT @ torch.matrix_power(M,10).mT @ (K.mT).mT
        # K_grad = (Q.mT @ prev_grad @ ((V * W)).mT @ torch.matrix_power(M,10).mT).mT
        # V_grad = (((Q @ K.mT @ torch.matrix_power(M,10)).mT @ prev_grad) * W)
        # M_grad = torch.zeros_like(M)
        # for i in range(10):
        #     M_grad += torch.matrix_power(M.mT, i) @ (Q @ K.mT).mT @ (prev_grad @ ((V * W)).mT) @ torch.matrix_power(M.mT, 10 - 1 - i)
        # W_grad = (V * ((Q @ K.mT @ torch.matrix_power(M,10)).mT @ prev_grad))
        
        
        # Q_grad = torch.matrix_power(M,10).mT @ prev_grad @ ((V * W)).mT @ (K.mT).mT
        # K_grad = ((torch.matrix_power(M,10) @ Q).mT @ prev_grad @ ((V * W)).mT).mT
        # V_grad = (((torch.matrix_power(M,10) @ Q @ K.mT).mT @ prev_grad) * W)
        # M_grad = torch.zeros_like(M)
        # for i in range(10):
        #     M_grad += torch.matrix_power(M.mT, 10 - 1 - i) @ (prev_grad @ ((V * W)).mT @ K) @ Q.mT @ torch.matrix_power(M.mT, i)
        # W_grad = (V * ((torch.matrix_power(M,10) @ Q @ K.mT).mT @ prev_grad))
        
        
        Q_grad = torch.matrix_power(M, 10).mT @ prev_grad @ ((V * W)).mT @ torch.matrix_power(M, 10).mT @ (K.mT).mT
        K_grad = ((torch.matrix_power(M, 10) @ Q).mT @ prev_grad @ ((V * W)).mT @ torch.matrix_power(M, 10).mT).mT
        V_grad = (((torch.matrix_power(M, 10) @ Q @ K.mT @ torch.matrix_power(M, 10)).mT @ prev_grad) * W)
        M_grad = torch.zeros_like(M)
        for i in range(10):
            M_grad += torch.matrix_power(M.mT, 10 - 1 - i) @ (prev_grad @ ((V * W)).mT @ torch.matrix_power(M, 10).mT @ K) @ Q.mT @ torch.matrix_power(M.mT, i)
        for i in range(10):
            M_grad += torch.matrix_power(M.mT, i) @ (torch.matrix_power(M, 10) @ Q @ K.mT).mT @ (prev_grad @ ((V * W)).mT) @ torch.matrix_power(M.mT, 10 - 1 - i)
        W_grad = (V * ((torch.matrix_power(M, 10) @ Q @ K.mT @ torch.matrix_power(M, 10)).mT @ prev_grad))
        
        
        return Q_grad, K_grad, V_grad, W_grad, M_grad
    
    

N, H, S, D = 1, 2, 16, 12
Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
K = torch.rand(N, H, S, D, requires_grad=True).cuda()
V = torch.rand(N, H, S, D, requires_grad=True).cuda()
W = torch.rand(N, H, S, D, requires_grad=True).cuda()
M = torch.rand(N, H, S, S, requires_grad=True).cuda()
torch.autograd.gradcheck(Function.apply, (Q.double(), K.double(), V.double(), W.double(), M.double()), eps=1e-4)