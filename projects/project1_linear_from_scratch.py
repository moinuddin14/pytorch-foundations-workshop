"""Project 1 — Build and test nn.Linear yourself.

Run from the repository root:
    python projects/project1_linear_from_scratch.py

Solution (open only after attempting the TODOs):
    projects/answers/project1_linear_from_scratch.py
"""
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    """Implement y = x @ W.T + b using registered parameters."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # TODO 1: create self.weight as nn.Parameter of shape (out_features, in_features)
        #         initialize with torch.randn(...) * 0.1
        # TODO 2: create self.bias as nn.Parameter of zeros (out_features) if bias else None
        pass

    def forward(self, x):
        # TODO 3: return x @ self.weight.T + self.bias  (mind the None bias case)
        pass


def main():
    torch.manual_seed(42)
    layer = MyLinear(4, 3)
    parameters = dict(layer.named_parameters())
    assert "weight" in parameters, "TODO 1: register self.weight with nn.Parameter"
    assert "bias" in parameters, "TODO 2: register self.bias with nn.Parameter"
    assert parameters["weight"].shape == (3, 4), "weight must be (out_features, in_features)"
    assert parameters["bias"].shape == (3,), "bias must contain one value per output"
    print("checkpoint 1 — parameter registration: PASS")

    # Sanity check against real nn.Linear
    real = nn.Linear(4, 3)
    with torch.no_grad():
        real.weight.copy_(layer.weight)
        real.bias.copy_(layer.bias)
    x = torch.randn(5, 4)
    output = layer(x)
    assert output is not None, "TODO 3: return the computed output from forward"
    assert output.shape == (5, 3)
    assert torch.allclose(output, real(x))
    print("checkpoint 2 — forward pass matches nn.Linear: PASS")

    # Train it on synthetic data: y = 2.5x + 1.0
    torch.manual_seed(3)
    layer = MyLinear(1, 1)
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.5 * x + 1.0 + 0.1 * torch.randn_like(x)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    for _ in range(300):
        opt.zero_grad()
        loss = loss_fn(layer(x), y)
        loss.backward()
        opt.step()
    w, b = layer.weight.item(), layer.bias.item()
    print(f"learned: y = {w:.3f}x + {b:.3f}  (true: 2.5, 1.0)")
    assert abs(w - 2.5) < 0.1 and abs(b - 1.0) < 0.1
    print("checkpoint 3 — custom layer trains: PASS")


if __name__ == "__main__":
    main()
