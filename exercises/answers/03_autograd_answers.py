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
