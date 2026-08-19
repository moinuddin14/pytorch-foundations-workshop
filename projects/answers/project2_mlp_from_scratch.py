"""Project 2 — Runnable XOR solution with implementation checks.

Build an MLP that solves XOR using ONLY nn.Module + nn.Parameter.
No nn.Linear, no nn.ReLU, no nn.Sequential, no nn.Softmax, no nn.CrossEntropyLoss.

The network is 2 -> 8 -> 2: two output units, one per class.
Training uses YOUR softmax + cross-entropy; predictions use argmax.

Run from the repository root:
    python projects/answers/project2_mlp_from_scratch.py
"""
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    """Reuse your Project 1 layer (or write it here)."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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


def relu(x):
    """Return the element-wise rectified linear activation."""
    return torch.clamp(x, min=0.0)


def softmax(logits):
    """Return stable row-wise class probabilities."""
    e = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return e / e.sum(dim=1, keepdim=True)


def cross_entropy(probs, targets_onehot):
    """Return mean cross-entropy for one-hot targets."""
    return -(targets_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()


class MyMLP(nn.Module):
    """A 2 -> hidden -> 2 network. Two MyLinear layers, ReLU between."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = MyLinear(in_dim, hidden_dim)
        self.fc2 = MyLinear(hidden_dim, out_dim)

    def forward(self, x):
        h = relu(self.fc1(x))
        return self.fc2(h)


def check_primitives():
    values = torch.tensor([[-2.0, 0.0, 3.0]])
    assert torch.equal(relu(values), torch.tensor([[0.0, 0.0, 3.0]]))
    logits = torch.tensor([[1.0, 2.0], [1000.0, 1001.0]])
    probabilities = softmax(logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    assert torch.isfinite(cross_entropy(probabilities, targets))


def main():
    check_primitives()
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 0])               # XOR truth table

    # one-hot targets: class 0 -> [1, 0], class 1 -> [0, 1]
    Y_onehot = torch.zeros(4, 2)
    Y_onehot[torch.arange(4), labels] = 1.0

    torch.manual_seed(0)
    model = MyMLP(2, 8, 2)                            # two output units: one per class
    assert model(X).shape == (4, 2)
    assert len(list(model.parameters())) == 4
    opt = torch.optim.SGD(model.parameters(), lr=0.5)

    for step in range(2000):
        opt.zero_grad()
        probs = softmax(model(X))                     # logits -> probabilities
        loss = cross_entropy(probs, Y_onehot)         # -log(p of the correct class)
        loss.backward()
        opt.step()
        if step in (0, 499, 1999):
            print(f"step {step:4d} loss={loss.item():.4f}")

    preds = model(X).argmax(dim=1).tolist()           # pick the class with the highest logit
    truth = labels.tolist()
    print("predictions:", preds)
    print("truth      :", truth)
    print("accuracy   :", sum(p == t for p, t in zip(preds, truth)), "/ 4")
    assert preds == truth, "XOR not solved — debug your forward/backward!"
    print("Project 2 solution: PASS")


if __name__ == "__main__":
    main()
