+++
title = "Project 1: Rebuild nn.Linear"
description = "Implement a registered linear layer, compare it with PyTorch, check gradients, and train it."
weight = 1
completable = true
notebook = "notebooks/labs/project1_linear_lab.ipynb"
+++
## Build before revealing

Complete TODOs 1-3 in the [starter](#start-project) before reading the code below.

Your complete specification is:

- `weight.shape == (out_features, in_features)`
- `bias.shape == (out_features,)`
- `forward(x) == x @ weight.T + bias`

Run the starter's checks after every TODO.

```python
import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # parameters — the ONLY trainable things in this layer
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)   # keep parameters() honest

    def forward(self, x):
        y = x @ self.weight.T          # (batch, in) @ (in, out) -> (batch, out)
        if self.bias is not None:
            y = y + self.bias          # broadcasting adds bias to every row
        return y

layer = MyLinear(4, 3)
print(layer)
print("weight shape:", layer.weight.shape, " bias shape:", layer.bias.shape)
```

{{< output >}}
MyLinear()
weight shape: torch.Size([3, 4])  bias shape: torch.Size([3])
{{< /output >}}

## Check 1: batch shape

Predict the output shape, run the cell, then add an assertion for it.

```python
x = torch.randn(5, 4)
out = layer(x)
print("input :", x.shape)
print("output:", out.shape)   # (5, 3)
assert out.shape == (5, 3)
```

{{< output >}}
input : torch.Size([5, 4])
output: torch.Size([5, 3])
{{< /output >}}

## Check 2: compare with `nn.Linear`

Copy identical weights into both implementations and assert their outputs match.

```python
torch.manual_seed(42)
mine = MyLinear(4, 3)

torch.manual_seed(42)
real = nn.Linear(4, 3)

# force identical weights (manual_seed makes randn identical, but scale differs — copy to be exact)
with torch.no_grad():
    real.weight.copy_(mine.weight)
    real.bias.copy_(mine.bias)

x = torch.randn(5, 4)
print("mine:", mine(x))
print("real:", real(x))
identical = torch.allclose(mine(x), real(x))
print("\nidentical?", identical)
assert identical
```

{{< output >}}
mine: tensor([[ 0.0235, -0.0925,  0.0191],
        [ 0.0590, -0.3120,  0.0898],
        [ 0.0181,  0.0568,  0.1097],
        [-0.0153,  0.6388, -0.0482],
        [-0.0125,  0.2759, -0.0085]], grad_fn=<AddBackward0>)
real: tensor([[ 0.0235, -0.0925,  0.0191],
        [ 0.0590, -0.3120,  0.0898],
        [ 0.0181,  0.0568,  0.1097],
        [-0.0153,  0.6388, -0.0482],
        [-0.0125,  0.2759, -0.0085]], grad_fn=<AddmmBackward0>)

identical? True
{{< /output >}}

## Check 3: compare gradients numerically

Reuse the central-difference checker from Module 3. The implementation passes only when both weight and bias gradients match.

```python
def gradcheck_layer(layer, x):
    """Compare autograd gradients vs central-difference for one weight."""
    loss = layer(x).pow(2).sum()
    loss.backward()

    eps = 1e-3   # float32 precision: 1e-6 loses too many digits (see Module 3.6)
    for name, p in layer.named_parameters():
        if p.grad is None:
            continue
        numeric = torch.zeros_like(p)
        for idx in range(p.numel()):
            flat = p.detach().flatten()
            flat[idx] += eps
            plus = layer(x).pow(2).sum()
            flat[idx] -= 2 * eps
            minus = layer(x).pow(2).sum()
            flat[idx] += eps
            numeric.flatten()[idx] = (plus - minus) / (2 * eps)
        ok = torch.allclose(p.grad, numeric, atol=1e-3)
        print(f"{name:8s}: max err = {(p.grad - numeric).abs().max().item():.2e}  -> {'PASS' if ok else 'FAIL'}")
        assert ok, f"gradient check failed for {name}"

torch.manual_seed(0)
layer = MyLinear(3, 2)
gradcheck_layer(layer, torch.randn(4, 3))
```

{{< output >}}
weight  : max err = 5.51e-06  -> PASS
bias    : max err = 2.53e-06  -> PASS
{{< /output >}}

## Check 4: train the custom layer

Fit `y = 2.5*x + 1`. Then change the target to `y = -3*x + 0.25` and verify the same implementation learns again.

```python
import matplotlib.pyplot as plt

torch.manual_seed(3)
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 2.5 * x + 1.0 + 0.1 * torch.randn_like(x)

model = MyLinear(1, 1)                 # our layer!
opt = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

losses = []
for _ in range(300):
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())

w, b = model.weight.item(), model.bias.item()
print(f"learned: y = {w:.3f} x + {b:.3f}   (true: y = 2.5 x + 1.0)")
assert abs(w - 2.5) < 0.1 and abs(b - 1.0) < 0.1

plt.figure(figsize=(8, 3))
plt.plot(losses)
plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training MyLinear (built from scratch)")
plt.show()
```

{{< output >}}
learned: y = 2.487 x + 1.008   (true: y = 2.5 x + 1.0)
/var/folders/fc/9slvrgbn7pg6n5rw2h1pst400000gn/T/ipykernel_13484/3503427985.py:27: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
{{< /output >}}

## Done when

Your layer has registered parameters, matches `nn.Linear`, passes gradient checks, and learns two different lines. That is the practical definition of a correct layer.


## Start project

Open the Colab scaffold above and complete the TODOs. If you prefer a local file, use the [Python starter]({{< param githubRepo >}}/blob/main/projects/project1_linear_from_scratch.py).

{{< details summary="Reveal the complete reference implementation" >}}

```python
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
```

{{< /details >}}
