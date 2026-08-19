+++
title = "What PyTorch actually does"
description = "Run one formula, inspect its gradient, then train a tiny model end to end."
weight = 1
completable = true
notebook = "notebooks/01_what_is_pytorch.ipynb"
+++
## Build first

Run the next three examples. Before each run, predict the output.

You only need three terms:

- **tensor:** an array of numbers;
- **autograd:** records tensor operations and computes gradients;
- **`nn`:** modules, losses, and optimizers used to build models.

The working loop is `prediction → loss → backward() → gradients → step()`.

## One implementation rule

You write Python such as `torch.matmul(a, b)`; PyTorch runs the heavy operation in compiled CPU/GPU kernels. Prefer tensor operations over Python element loops. You will benchmark this in Module 2.

## Lab 1: run the same formula twice

Run the NumPy cell. It computes the value but does not retain a differentiable history.

```python
import numpy as np

x = np.array([2.0])
y = x * x + 3.0
print("y =", y)
print("gradient?", getattr(y, "grad", "nope, NumPy does not record history"))
```

{{< output >}}
y = [7.]
gradient? nope, NumPy does not record history
{{< /output >}}

Now run the PyTorch version. Before `backward()`, predict `x.grad` for `y = x² + 3` at `x = 2`. Then change `x` to `3` and verify the new gradient.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x * x + 3.0
print("y =", y)
y.backward()          # <<< this is the difference
print("x.grad =", x.grad)   # dy/dx at x=2  →  2*x = 4
assert torch.equal(x.grad, torch.tensor([4.0]))
```

{{< output >}}
y = tensor([7.], grad_fn=<AddBackward0>)
x.grad = tensor([4.])
{{< /output >}}

## Lab 2: run a complete training loop

Run the cell once. Then change the generated data to `y = -2 * x + 1` and rerun. The learned weight should approach `-2` and the bias should approach `1`.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# data
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = 3 * x + 0.5 + 0.1 * torch.randn_like(x)

# model, loss, optimizer
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(200):
    pred = model(x)              # forward
    loss = loss_fn(pred, y)      # loss
    optimizer.zero_grad()        # reset old gradients
    loss.backward()              # backward: compute gradients
    optimizer.step()             # update weights

w, b = model.weight.item(), model.bias.item()
print(f"learned: y = {w:.3f} * x + {b:.3f}   (true: y = 3.0 * x + 0.5)")
assert abs(w - 3.0) < 0.05 and abs(b - 0.5) < 0.05
```

{{< output >}}
learned: y = 3.000 * x + 0.504   (true: y = 3.0 * x + 0.5)
{{< /output >}}

## Practical checkpoint

- Make a tracked scalar `x`, compute `y = 3*x**2`, and verify the gradient by hand.
- Point to the five operations in the training cell: forward, loss, zero gradients, backward, step.
- Explain the practical difference between the NumPy and PyTorch examples in one sentence.
