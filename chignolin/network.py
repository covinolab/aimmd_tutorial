import torch

class Network(torch.nn.Module):
    """
    Pytorch neural network.
    Input features size: 2 (x, y).
    Output size: 1 (logit of the committor).
    """
    def __init__(self):
        super().__init__()
        self.call_kwargs = {}
        # layers & activations
        n = 512
        self.input = torch.nn.Linear(2064, n)
        self.layer1 = torch.nn.Linear(n, n)
        self.layer2 = torch.nn.Linear(n, n)
        self.output = torch.nn.Linear(n, 1)
        self.activation1 = torch.nn.PReLU(n)
        self.activation2 = torch.nn.PReLU(n)
        self.reset_parameters()
    def forward(self, x):
        x = self.input(x)
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.output(x)
        return x
    def reset_parameters(self):
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()
        self.output.reset_parameters()
 
