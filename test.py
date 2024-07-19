import torch
import os
from main import main




tests = [
    {
        "function": "f(f(f((f(X @ A) + X) @ B) @ C) @ D)",
        "sequence": "f(f(f((f(X @ A) + X) @ B) @ C) @ D)",
        "shapes": {
            "X": ["B", "d"],
            "A": ["d", "d"],
            "B": ["d", "D"],
            "C": ["D", "D"],
            "D": ["D", "d"],
        },
        "function_types": {
            "f": "scalar"
        },
        "function_replace": {
            "f": "torch.nn.functional.silu",
            "f_der": "lambda x: torch.sigmoid(x) * (1 + x * (1 - torch.sigmoid(x)))"
        }
    },
    {
        "function": "softmax(((X @ Q) @ (X @ K).mT) * M, dim=-1) @ (X @ V)",
        "sequence": "softmax(((X Q) (X K)^T) * M) (X V)",
        "shapes": {
            "X": ["N", "S", "d"],
            "Q": ["d", "d_k"],
            "K": ["d", "d_k"],
            "V": ["d", "d_v"],
            "M": ["N", "S", "S"],
        },
        "function_types": {
            "softmax": "vector",
        },
        "function_replace": {
            "softmax": "torch.nn.functional.softmax",
            "Jacobian": "lambda x: torch.diag_embed(x) - x[..., :, None] * x[..., None, :]"
        }
    }
]


spaces = " "*8

def test():
    for test in tests:
        # Get the params
        function, sequence, shapes, function_types, function_replace = test["function"], test["sequence"], test["shapes"], test["function_types"], test["function_replace"]
        wanted_grads = list(shapes.keys())
        
        # Run the script
        matrices_and_functions = main(sequence, shapes, wanted_grads, function_types)
        
        gradients = {}
        for i,j in matrices_and_functions.items():
            if i not in wanted_grads:
                continue
            j = "+".join([g.torch_str() for g in j.grad])
            gradients[i] = j.replace("dL", "prev_grad")
        
        inputs = ", ".join(wanted_grads)
        inputs_double = ".double(), ".join(wanted_grads) + ".double()"
        inputs_grad = "_grad, ".join(wanted_grads) + "_grad"
        functions = f"\n".join([f"{i} = {j}" for i,j in function_replace.items()])
        gradients = f"\n{spaces}".join([f"{i}_grad = {j}" for i,j in gradients.items()])
        
        all_shapes = set()
        shape_values = {}
        for shape in shapes.values():
            for s in shape:
                all_shapes.add(s)
        for shape in all_shapes:
            shape_values[shape] = torch.randint(5, 25, (1,)).item()
        shapes_ = ", ".join(shape_values.keys()) + " = " + ", ".join([str(i) for i in shape_values.values()])
        
        tensor_shapes = {}
        for g in wanted_grads:
            s = ", ".join(shapes[g])
            tensor_shapes[g] = f"{g} = torch.rand({s}, requires_grad=True).cuda()"
        tensor_shapes = "\n".join([f"{i} = {j}" for i,j in tensor_shapes.items()])
        
        s = f"""
import torch

{functions}

class Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, {inputs}):
        ctx.save_for_backward({inputs})
        return {function}

    @staticmethod
    def backward(ctx, prev_grad):
        {inputs} = ctx.saved_tensors
        
        {gradients}
        
        return {inputs_grad}
    
    
{shapes_}
{tensor_shapes}


def test():
    torch.autograd.gradcheck(Function.apply, ({inputs_double}), eps=1e-4)
""".strip()


        # Write this massive string to a python file
        with open("tmp.py", "w") as f:
            f.write(s)
            
        # Import the test
        from tmp import test
        
        # Test the gradients
        passed = True
        try:
            test()
            print("Test passed")
        except:
            passed = False
            print("Test failed")
        
        # Delete the file
        del test
        os.remove("tmp.py")
    


if __name__ == "__main__":
    test()






