+++
title = "The training loop"
description = "Write SGD from memory, compare learning rates, batch data, and validate a model."
weight = 5
completable = true
notebook = "notebooks/05_the_training_loop.ipynb"
+++
## Lab 1: write the five steps from memory

Before revealing the snippet, write: **forward -> loss -> zero_grad -> backward -> step**. Then compare your code with this loop:

```python
for epoch in range(num_epochs):
    # 1. forward   — compute predictions
    pred = model(x)
    # 2. loss      — measure how wrong we are
    loss = loss_fn(pred, y)
    # 3. zero_grad — clear last batch's gradients (they accumulate!)
    optimizer.zero_grad()
    # 4. backward  — compute gradients via autograd
    loss.backward()
    # 5. step      — update the weights using the gradients
    optimizer.step()
```

`zero_grad()` clears accumulated gradients; `backward()` creates fresh ones; `step()` consumes them.

## Lab 2: reimplement one SGD step

Compute `manual = old_parameter - lr * gradient`, run `optimizer.step()`, and assert the values match.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(1, 1)
x = torch.randn(8, 1)
y = 3 * x + 0.5

# --- copy of weights before step ---
w0 = model.weight.detach().clone()
b0 = model.bias.detach().clone()

loss = (model(x) - y).pow(2).mean()
loss.backward()
grad_w = model.weight.grad.clone()
grad_b = model.bias.grad.clone()

# --- manual SGD ---
lr = 0.1
with torch.no_grad():
    w_manual = w0 - lr * grad_w
    b_manual = b0 - lr * grad_b

# --- optimizer SGD ---
opt = torch.optim.SGD(model.parameters(), lr=lr)
opt.step()

print("manual  w =", w_manual.item(), " b =", b_manual.item())
print("optim   w =", model.weight.item(), " b =", model.bias.item())
identical = torch.allclose(w_manual, model.weight) and torch.allclose(b_manual, model.bias)
print("identical?", identical)
assert identical
```

{{< output >}}
manual  w = 0.40212708711624146  b = 0.4576661288738251
optim   w = 0.40212708711624146  b = 0.4576661288738251
identical? True
{{< /output >}}

For plain SGD, `step()` applies `parameter -= lr * parameter.grad`. Other optimizers transform the gradient before applying an update.

## Lab 3: train the same model with three learning rates

Run all three experiments. Record the final loss and learned `(weight, bias)` for each rate.

```python
import matplotlib.pyplot as plt

def train(lr, epochs=300):
    torch.manual_seed(1)
    x = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = 3 * x + 0.5 + 0.1 * torch.randn_like(x)

    model = nn.Linear(1, 1)
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, model

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, lr in zip(axes, [0.001, 0.1, 5.0]):
    losses, model = train(lr)
    w, b = model.weight.item(), model.bias.item()
    ax.plot(losses)
    ax.set_title(f"lr={lr}  →  y = {w:.2f}x + {b:.2f}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 2)
plt.tight_layout()
plt.show()
print("true line: y = 3.00x + 0.50")
```

{{< output >}}
true line: y = 3.00x + 0.50
/var/folders/fc/9slvrgbn7pg6n5rw2h1pst400000gn/T/ipykernel_13480/795767836.py:32: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
{{< /output >}}

Interpret the evidence: `0.001` is slow, `0.1` converges, and `5.0` diverges. Add `lr=0.01` and place it on the spectrum.

## Lab 4: train from a `DataLoader`

Run the batched loop. Change `batch_size` from `32` to `400`, count optimizer steps per epoch, and compare convergence.

```python
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(2)
X = torch.linspace(-1, 1, 400).reshape(-1, 1)
Y = 2 * X + 1 + 0.05 * torch.randn_like(X)

dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(20):
    epoch_loss = 0.0
    for xb, yb in loader:            # one batch
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  loss={epoch_loss/len(loader):.4f}")

w, b = model.weight.item(), model.bias.item()
print(f"\nlearned: y = {w:.3f}x + {b:.3f}   (true: 2.0, 1.0)")
assert abs(w - 2.0) < 0.05 and abs(b - 1.0) < 0.05
```

{{< output >}}
epoch  0  loss=1.4292
epoch  5  loss=0.0179
epoch 10  loss=0.0026
epoch 15  loss=0.0024

learned: y = 1.997x + 1.001   (true: 2.0, 1.0)
{{< /output >}}

An **epoch** is one pass over the dataset; a **batch** is one chunk; a **step** is one parameter update. Print all three counts for this run.

## Lab 5: add validation

Run validation without changing parameters. Confirm the output has no `grad_fn`.

```python
model.eval()
with torch.no_grad():
    pred = model(X)
    val_loss = loss_fn(pred, Y)
print(f"final loss (eval mode): {val_loss.item():.4f}")
print("eval() + no_grad() — deterministic, no graph built")
assert pred.grad_fn is None
```

{{< output >}}
final loss (eval mode): 0.0024
eval() + no_grad() — deterministic, no graph built
{{< /output >}}

`eval()` changes Dropout/BatchNorm behavior; `no_grad()` disables graph recording. Use both.

## Practical checkpoint

From a blank cell, train `nn.Linear(1, 1)` on `y = 4*x - 2`, print loss every 25 steps, and report the learned parameters.
