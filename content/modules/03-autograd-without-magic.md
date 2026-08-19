+++
title = "Autograd without magic"
description = "Predict gradients, inspect the graph, update a parameter, and debug accumulation."
weight = 3
completable = true
notebook = "notebooks/03_autograd_without_magic.ipynb"
+++
## Lab 1: inspect a computation graph

PyTorch records operations that lead back to a tensor with `requires_grad=True`. Run the first cell and inspect `grad_fn` on the inputs and intermediate results. Leaves have `grad_fn=None`; operation results do not.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)   # leaf
w = torch.tensor(3.0, requires_grad=True)   # leaf
b = torch.tensor(1.0)                       # NOT a leaf (no requires_grad)

z = x * w        # MulBackward
y = z + b        # AddBackward

print("x.grad_fn:", x.grad_fn)
print("z.grad_fn:", z.grad_fn)
print("y.grad_fn:", y.grad_fn)
print("y =", y.item())
```

{{< output >}}
x.grad_fn: None
z.grad_fn: <MulBackward0 object at 0x106ad3490>
y.grad_fn: <AddBackward0 object at 0x15a8b4190>
y = 7.0
{{< /output >}}

Modify `b` to use `requires_grad=True`, rerun the forward pass, and later verify `b.grad == 1`.

## Lab 2: predict gradients before PyTorch

For `y = x*w + b`, write `dy/dx` and `dy/dw` on paper. Then call `backward()` and compare. `backward()` walks recorded operations in reverse and applies the chain rule.

```python
y.backward()
print("dy/dx =", x.grad.item(), "  (expected 3)")
print("dy/dw =", w.grad.item(), "  (expected 2)")
assert x.grad.item() == 3.0 and w.grad.item() == 2.0
```

{{< output >}}
dy/dx = 3.0   (expected 3)
dy/dw = 2.0   (expected 2)
{{< /output >}}

Change the formula to `y = (x*w + b)**2`. Predict all three gradients at the current values, then rerun.

## Lab 3: perform one manual training step

Run the linear-regression cell. Print the scalar loss, the gradients of `w` and `b`, and their values before and after the update.

```python
# data: y = 2x + 1 (with a little noise)
torch.manual_seed(0)
x = torch.linspace(-1, 1, 20).reshape(-1, 1)
y = 2 * x + 1 + 0.05 * torch.randn_like(x)

# parameters we'll learn
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

pred = x @ w + b                       # forward: matrix multiply + bias
loss = (pred - y).pow(2).mean()        # MSE
print("loss:", loss.item())

loss.backward()
print("d(loss)/dw:", w.grad.item())
print("d(loss)/db:", b.grad.item())

# one gradient descent step, by hand:
lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad
print("after one step: w =", w.item(), " b =", b.item(), " (true: 2.0, 1.0)")
```

{{< output >}}
loss: 9.570019721984863
d(loss)/dw: -2.1665492057800293
d(loss)/db: -5.052918434143066
after one step: w = -0.6753404140472412  b = -1.0038158893585205  (true: 2.0, 1.0)
{{< /output >}}

Only tracked leaves receive `.grad`. The weight update uses `torch.no_grad()` because the update itself is not part of the model's forward computation.

## Lab 4: reproduce the accumulation bug

Call `backward()` on two fresh forward graphs without clearing `.grad`. Observe the sum, then fix it with `x.grad.zero_()`. Optimizers provide `zero_grad()` for all parameters at once.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print("after 1st backward, grad =", x.grad.item())   # 2*x = 4

y2 = x ** 2
y2.backward()
print("after 2nd backward, grad =", x.grad.item())   # 4 + 4 = 8  (accumulated!)

x.grad.zero_()   # or optimizer.zero_grad()
print("after zero_, grad =", x.grad.item())
assert x.grad.item() == 0.0
```

{{< output >}}
after 1st backward, grad = 4.0
after 2nd backward, grad = 8.0
after zero_, grad = 0.0
{{< /output >}}

## Lab 5: turn graph recording off

Run the same operation with and without `torch.no_grad()`. Compare `grad_fn`. Use this pattern for validation and inference.

```python
x = torch.randn(1000, 1000, requires_grad=True)

# with tracking:
y = (x * x).sum()
print("tracked: y.grad_fn =", y.grad_fn.__class__.__name__)

# without tracking:
with torch.no_grad():
    y2 = (x * x).sum()
print("no_grad: y2.grad_fn =", y2.grad_fn, " (None — nothing recorded)")
```

{{< output >}}
tracked: y.grad_fn = SumBackward0
no_grad: y2.grad_fn = None  (None — nothing recorded)
{{< /output >}}

## Lab 6: implement a numerical gradient check

Compare autograd with `(f(x+eps) - f(x-eps)) / (2*eps)`. Run it at three values of `x`; use `eps=1e-3` for float32.

```python
def f(x):
    return torch.sin(x) * x ** 2 + 3 * x

x0 = torch.tensor(1.7, requires_grad=True)
f(x0).backward()
analytic = x0.grad.item()

eps = 1e-3   # float32 precision: 1e-6 loses too many digits (try it!)
with torch.no_grad():
    # fresh detached values — NOT x0 itself, or the graph gets polluted
    numeric = (f(x0.detach() + eps) - f(x0.detach() - eps)) / (2 * eps)

print(f"autograd:  {analytic:.8f}")
print(f"numeric:   {numeric.item():.8f}")
matches = abs(analytic - numeric.item()) < 1e-3
print(f"match?     {matches}")
assert matches
```

{{< output >}}
autograd:  5.99929953
numeric:   5.99956465
match?     True
{{< /output >}}

Keep this checker: you will reuse it on your custom layer in Project 1.

## Minimal model

`loss.backward()` walks the current graph once, applies local derivatives, and accumulates results in tracked leaves. The graph is normally freed afterwards, so rerun the forward pass before another backward pass.

## Practical checkpoint

Implement two gradient-descent steps for a scalar parameter without an optimizer. Each step must compute a fresh loss, clear the old gradient, call `backward()`, and update inside `no_grad()`.
