+++
title = "Autograd implementation lab"
description = "Predict, inspect, break, fix, and numerically verify gradients."
weight = 2
completable = true
notebook = "notebooks/labs/03_autograd_lab.ipynb"
+++
Write code only after calculating each derivative by hand.
## 1. Predict and verify a gradient

For `x = torch.tensor(3.0, requires_grad=True)`, compute `z = 2 * x**3`. Write `dz/dx` on paper, call `backward()`, and assert the values match.

## 2. Inspect a graph

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
w = torch.tensor([3.0, 4.0], requires_grad=True)
y = (x * w).sum()
```

- Print the operation classes stored in `grad_fn`.
- Predict `x.grad` and `w.grad`, then assert both after `backward()`.

## 3. Reproduce and fix accumulation

Call `(x**2).backward()` five times for `x = 2`. Assert the accumulated gradient is `20`, clear it, and assert it is `0`.

Then train a scalar parameter for ten steps twice: once correctly and once without clearing gradients. Compare the final losses.

## 4. Compare `no_grad()` and `detach()`

Create one result inside `torch.no_grad()` and another with `.detach()`. Assert that neither tracks gradients. Use `no_grad()` around a manual weight update.

## 5. Implement `gradcheck`

Write `gradcheck(f, x0, eps=1e-3)` using central differences for `f(x) = sin(x) * x**2`. Test `0.5`, `1.3`, and `-2.0`; assert every error is below `1e-3`.

Done means every expected gradient is encoded as an assertion, not just printed.


{{< details summary="Reveal the runnable solution" >}}

```python
"""Runnable solutions for Module 3. Run: python exercises/answers/03_autograd_answers.py"""

import torch


def predicted_gradient():
    x = torch.tensor(3.0, requires_grad=True)
    z = 2 * x**3
    z.backward()
    assert x.grad.item() == 54.0


def graph_forensics():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    w = torch.tensor([3.0, 4.0], requires_grad=True)
    y = (x * w).sum()
    y.backward()
    assert torch.equal(x.grad, w.detach())
    assert torch.equal(w.grad, x.detach())


def accumulation():
    x = torch.tensor(2.0, requires_grad=True)
    for _ in range(5):
        (x**2).backward()
    assert x.grad.item() == 20.0
    x.grad.zero_()
    assert x.grad.item() == 0.0


def no_grad_and_detach():
    x = torch.tensor([1.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    z = (x * 2).detach()
    assert not y.requires_grad and not z.requires_grad


def f(x):
    return torch.sin(x) * x**2


def gradcheck(function, x0, eps=1e-3):
    x = torch.tensor(x0, requires_grad=True)
    function(x).backward()
    with torch.no_grad():
        numeric = (function(x.detach() + eps) - function(x.detach() - eps)) / (2 * eps)
    return abs(x.grad.item() - numeric.item()) < 1e-3


if __name__ == "__main__":
    predicted_gradient()
    graph_forensics()
    accumulation()
    no_grad_and_detach()
    assert all(gradcheck(f, x0) for x0 in (0.5, 1.3, -2.0))
    print("Module 3 solutions: PASS")
```

{{< /details >}}
