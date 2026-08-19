"""Project 2 — Build and test an XOR network without high-level layers.

Build an MLP that solves XOR using ONLY nn.Module + nn.Parameter.
No nn.Linear, no nn.ReLU, no nn.Sequential, no nn.Softmax, no nn.CrossEntropyLoss.

The network is 2 -> 8 -> 2: two output units, one per class.
Training uses YOUR softmax + cross-entropy; predictions use argmax.

Run from the repository root:
    python projects/project2_mlp_from_scratch.py

Solution (open only after attempting the TODOs):
    projects/answers/project2_mlp_from_scratch.py
"""
import torch
import torch.nn as nn


class MyLinear(nn.Module):
    """Reuse your Project 1 layer (or write it here)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        # TODO 1: weight (out, in) * 0.1, bias zeros
        pass

    def forward(self, x):
        # TODO 2: x @ W.T + b
        pass


def relu(x):
    """TODO 3: ReLU = element-wise max(x, 0). One line. (torch.clamp or torch.maximum)"""
    pass


def softmax(logits):
    """TODO 4: stable softmax — subtract row max before exp, then normalize rows."""
    pass


def cross_entropy(probs, targets_onehot):
    """TODO 5: -mean( sum over classes of target * log(prob) )"""
    pass


class MyMLP(nn.Module):
    """A 2 -> hidden -> 2 network. Two MyLinear layers, ReLU between."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # TODO 6: self.fc1 = MyLinear(in_dim, hidden_dim); self.fc2 = MyLinear(hidden_dim, out_dim)
        pass

    def forward(self, x):
        # TODO 7: h = relu(fc1(x)); return fc2(h)
        pass


def check_primitives():
    values = torch.tensor([[-2.0, 0.0, 3.0]])
    relu_output = relu(values)
    assert relu_output is not None, "TODO 3: return the ReLU result"
    assert torch.equal(relu_output, torch.tensor([[0.0, 0.0, 3.0]]))

    logits = torch.tensor([[1.0, 2.0], [1000.0, 1001.0]])
    probabilities = softmax(logits)
    assert probabilities is not None, "TODO 4: return softmax probabilities"
    assert probabilities.shape == logits.shape
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))

    targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    loss = cross_entropy(probabilities, targets)
    assert loss is not None, "TODO 5: return the cross-entropy loss"
    assert torch.isfinite(loss)
    print("checkpoint 1 — activation and loss functions: PASS")


def main():
    check_primitives()
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    labels = torch.tensor([0, 1, 1, 0])               # XOR truth table

    # one-hot targets: class 0 -> [1, 0], class 1 -> [0, 1]
    Y_onehot = torch.zeros(4, 2)
    Y_onehot[torch.arange(4), labels] = 1.0

    torch.manual_seed(0)
    model = MyMLP(2, 8, 2)                            # two output units: one per class
    logits = model(X)
    assert logits is not None, "TODOs 6-7: construct the layers and return logits"
    assert logits.shape == (4, 2)
    assert len(list(model.parameters())) == 4
    print("checkpoint 2 — MLP shape and parameter registration: PASS")
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
    print("checkpoint 3 — XOR solved: PASS")


if __name__ == "__main__":
    main()
