+++
title = "Project 2: Solve XOR with your MLP"
description = "Implement activations and loss, compose a two-layer network, and prove why non-linearity matters."
weight = 2
completable = true
notebook = "notebooks/labs/project2_xor_lab.ipynb"
+++
## Build before revealing

Use the [starter](#start-project). First implement `relu`, stable `softmax`, and `cross_entropy`; run their checks. Then complete the two custom linear layers.

The task is concrete: make predictions `[0, 1, 1, 0]` for the four XOR inputs without using high-level layers. Start by running the linear baseline below and observing its failure.

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])   # XOR truth table

# First, prove a single linear layer CANNOT solve XOR:
torch.manual_seed(0)
lin = nn.Linear(2, 1)
opt = torch.optim.SGD(lin.parameters(), lr=0.5)
loss_fn = nn.MSELoss()

losses = []
for _ in range(2000):
    opt.zero_grad()
    loss = loss_fn(lin(X), Y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

preds = (lin(X) > 0.5).int().flatten().tolist()
truth = Y.int().flatten().tolist()
print("linear model final loss:", f"{losses[-1]:.4f}")
print("predictions:", preds)
print("truth      :", truth)
linear_correct = sum(p == t for p, t in zip(preds, truth))
print("accuracy   :", linear_correct, "/ 4  ← stuck (linear can't bend)")
assert linear_correct < 4
```

{{< output >}}
linear model final loss: 0.2500
predictions: [0, 0, 0, 0]
truth      : [0, 1, 1, 0]
accuracy   : 2 / 4  ← stuck (linear can't bend)
{{< /output >}}

## Check 1: implement the four ingredients

Run each function separately. Assert ReLU has no negative output, softmax rows sum to one, and cross-entropy is lower for better predictions.

```python
# --- our layer from Module 6 ---
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

# --- non-linearity: ReLU(x) = max(x, 0) ---
def relu(x):
    return torch.clamp(x, min=0.0)   # element-wise max with 0

# --- softmax: turns logits into probabilities ---
def softmax(logits):
    e = torch.exp(logits - logits.max(dim=1, keepdim=True).values)  # subtract max for stability
    return e / e.sum(dim=1, keepdim=True)

# --- cross-entropy loss on one-hot targets: -mean(log p_correct) ---
def cross_entropy(probs, targets_onehot):
    return -(targets_onehot * torch.log(probs + 1e-9)).sum(dim=1).mean()

# quick sanity check
logits = torch.tensor([[2.0, 1.0, 0.1]])
print("softmax sums to 1:", softmax(logits).sum().item())
probabilities = softmax(logits)
assert torch.allclose(probabilities.sum(dim=1), torch.ones(1))
print("argmax preserved:", probabilities.argmax().item() == logits.argmax().item())
```

{{< output >}}
softmax sums to 1: 1.0000001192092896
argmax preserved: True
{{< /output >}}

## Check 2: compose the MLP

Implement `2 -> 8 -> 2`: `fc1`, ReLU, then `fc2`. Pass a `(4, 2)` batch and assert the logits have shape `(4, 2)`.

```python
class MyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = MyLinear(in_dim, hidden_dim)   # our layer!
        self.fc2 = MyLinear(hidden_dim, out_dim)  # our layer!

    def forward(self, x):
        h = relu(self.fc1(x))      # hidden activations
        return self.fc2(h)         # logits

model = MyMLP(2, 8, 2)
print(model)
print("\nparameters:")
for name, p in model.named_parameters():
    print(f"  {name:8s} shape={tuple(p.shape)}")
assert model(X).shape == (4, 2)
assert len(list(model.parameters())) == 4
```

{{< output >}}
MyMLP(
  (fc1): MyLinear()
  (fc2): MyLinear()
)

parameters:
  fc1.weight shape=(8, 2)
  fc1.bias shape=(8,)
  fc2.weight shape=(2, 8)
  fc2.bias shape=(2,)
{{< /output >}}

## Check 3: train and classify

Run the same five-step loop used earlier. Convert labels to one-hot targets, train on cross-entropy, and predict with `argmax`.

```python
torch.manual_seed(0)
model = MyMLP(2, 8, 2)                     # two output units: one per class

# one-hot targets: class 0 -> [1, 0], class 1 -> [0, 1]
labels = Y.flatten().long()
Y_onehot = torch.zeros(4, 2)
Y_onehot[torch.arange(4), labels] = 1.0
print("one-hot targets:\n", Y_onehot)

opt = torch.optim.SGD(model.parameters(), lr=0.5)

losses = []
for _ in range(2000):
    opt.zero_grad()
    probs = softmax(model(X))              # logits -> probabilities
    loss = cross_entropy(probs, Y_onehot)  # -log(probability of the correct class)
    loss.backward()
    opt.step()
    losses.append(loss.item())

preds = model(X).argmax(dim=1).tolist()    # pick the class with the highest logit
truth = labels.tolist()
print("\nMLP final loss:", f"{losses[-1]:.4f}")
print("predictions:", preds)
print("truth      :", truth)
correct = sum(p == t for p, t in zip(preds, truth))
print("accuracy   :", correct, "/ 4  ← solved!")
assert preds == truth
```

{{< output >}}
one-hot targets:
 tensor([[1., 0.],
        [0., 1.],
        [0., 1.],
        [1., 0.]])
MLP final loss: 0.0003
predictions: [0, 1, 1, 0]
truth      : [0, 1, 1, 0]
accuracy   : 4 / 4  ← solved!
{{< /output >}}

The linear baseline reaches only 2/4; the MLP reaches 4/4. Now plot the learned class probability over a grid.

```python
xs = torch.linspace(-0.5, 1.5, 120)
ys = torch.linspace(-0.5, 1.5, 120)
XX, YY = torch.meshgrid(xs, ys, indexing="ij")          # every (x1, x2) point in the plane
points = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)

with torch.no_grad():
    probs = softmax(model(points))                      # (14400, 2) — one probability pair per point
    ZZ = probs[:, 1].reshape(120, 120)                  # probability of class 1

plt.figure(figsize=(5.5, 4.8))
plt.contourf(XX, YY, ZZ, levels=20, cmap="coolwarm")
plt.colorbar(label="P(class 1)")
plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c="red",   s=120, marker="o", edgecolors="white", label="class 1")
plt.scatter(X[Y.flatten() == 0, 0], X[Y.flatten() == 0, 1], c="blue",  s=120, marker="s", edgecolors="white", label="class 0")
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("P(class 1) — learned by our hand-built MLP")
plt.legend()
plt.show()
```

{{< output >}}
/var/folders/fc/9slvrgbn7pg6n5rw2h1pst400000gn/T/ipykernel_13489/2977678034.py:18: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
{{< /output >}}

Remove ReLU, retrain, and observe the failure. This experiment isolates the value of the non-linearity.

## Stretch implementation: three classes

Generate three 2-D clusters, change the model to `MyMLP(2, 8, 3)`, create three-wide one-hot targets, and reuse the same loop. Done means at least 90% training accuracy.

## Final practical checkpoint

From a blank file, rebuild `MyLinear`, ReLU, softmax, cross-entropy, the MLP, and its training loop. Keep each function testable in isolation.

Continue with the [exercise sets](../../practice/) and keep the [cheatsheet](../../reference/cheatsheet/) nearby.


## Start project

Open the Colab scaffold above and complete the TODOs. If you prefer a local file, use the [Python starter]({{< param githubRepo >}}/blob/main/projects/project2_mlp_from_scratch.py).

{{< details summary="Reveal the complete reference implementation" >}}

```python
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
```

{{< /details >}}
