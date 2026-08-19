"""Project 1 — Runnable solution with implementation checks.

Run from the repository root:
    python projects/answers/project1_linear_from_scratch.py
"""
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    """Implement y = x @ W.T + b using registered parameters."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # PyTorch stores Linear weights as (out_features, in_features).
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


def main():
    torch.manual_seed(42)
    layer = MyLinear(4, 3)
    parameters = dict(layer.named_parameters())
    assert parameters["weight"].shape == (3, 4)
    assert parameters["bias"].shape == (3,)

    # Sanity check against real nn.Linear
    real = nn.Linear(4, 3)
    with torch.no_grad():
        real.weight.copy_(layer.weight)
        real.bias.copy_(layer.bias)
    x = torch.randn(5, 4)
    assert torch.allclose(layer(x), real(x))

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
    print("Project 1 solution: PASS")


if __name__ == "__main__":
    main()
