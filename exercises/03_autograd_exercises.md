# Module 3 — Autograd implementation lab

Write code only after calculating each derivative by hand. Runnable solutions: `python exercises/answers/03_autograd_answers.py`.

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
