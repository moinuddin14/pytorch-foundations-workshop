+++
title = "Autograd: how PyTorch learns"
description = "See why gradients exist, use them to improve a prediction, and train a line from scratch."
weight = 3
completable = true
notebook = "notebooks/03_autograd_without_magic.ipynb"
+++
## Why does autograd exist?

A model makes a prediction. A **loss** says how wrong that prediction is. To improve the model, we need to know how each weight contributed to the loss.

**Autograd calculates that contribution for every trainable value.** The result is called a **gradient**. An optimizer then uses the gradient to update the value.

By the end of this notebook, you will use autograd to make a bad prediction better and train a line from scratch.

## The whole idea in four moves

1. **Forward:** make a prediction.
2. **Loss:** measure the error.
3. **Backward:** ask autograd for gradients with `loss.backward()`.
4. **Update:** move each trainable value in the direction that reduces the loss.

That is the intended use of autograd. We will build these four moves one at a time.

```python
import torch

print("PyTorch version:", torch.__version__)
```

{{< output >}}
PyTorch version: 2.1.2
{{< /output >}}

## Example 1 — Ask for one simple gradient

Start with `y = x²` at `x = 3`. The slope is `dy/dx = 2x`, so the expected gradient is `6`.

`requires_grad=True` tells PyTorch: **remember the operations that use this value because I will ask for a gradient later.**

```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2

print("y =", y.item())
print("x.grad before backward =", x.grad)

y.backward()

print("x.grad after backward  =", x.grad.item())
assert x.grad.item() == 6.0
```

{{< output >}}
y = 9.0
x.grad before backward = None
x.grad after backward  = 6.0
{{< /output >}}

What happened?

- The forward calculation produced `y = 9`.
- PyTorch remembered that `y` came from `x ** 2`.
- `y.backward()` worked backward through that calculation.
- The answer was stored in `x.grad`.

**Try it:** change `x` to `4`. Predict the gradient before rerunning the cell.

## Example 2 — Use gradients to improve a prediction

Now use autograd for its real purpose. Our tiny model is:

`prediction = weight × input + bias`

For input `2`, the target answer is `10`. We start with `weight = 1` and `bias = 0`, so the first prediction is only `2`.

The input and target are data, so they do not need gradients. The weight and bias are trainable, so they use `requires_grad=True`.

```python
input_value = torch.tensor(2.0)
target = torch.tensor(10.0)

weight = torch.tensor(1.0, requires_grad=True)
bias = torch.tensor(0.0, requires_grad=True)

# 1. Forward: make a prediction
prediction = weight * input_value + bias

# 2. Loss: square the prediction error
loss = (prediction - target) ** 2

# 3. Backward: calculate gradients for weight and bias
loss.backward()

print(f"prediction before update: {prediction.item():.1f}")
print(f"loss before update:       {loss.item():.1f}")
print(f"weight gradient:          {weight.grad.item():.1f}")
print(f"bias gradient:            {bias.grad.item():.1f}")

assert weight.grad.item() == -32.0
assert bias.grad.item() == -16.0

# 4. Update: subtract a small amount of each gradient
learning_rate = 0.1
with torch.no_grad():
    weight -= learning_rate * weight.grad
    bias -= learning_rate * bias.grad

new_prediction = weight * input_value + bias
new_loss = (new_prediction - target) ** 2

print(f"prediction after update:  {new_prediction.item():.1f}")
print(f"loss after update:        {new_loss.item():.1f}")
assert new_loss.item() < loss.item()
```

{{< output >}}
prediction before update: 2.0
loss before update:       64.0
weight gradient:          -32.0
bias gradient:            -16.0
prediction after update:  10.0
loss after update:        0.0
{{< /output >}}

### How to read a gradient

Both gradients were negative. Our update subtracts the gradient, so subtracting a negative number **increased** the weight and bias. That moved the prediction from `2` toward `10`.

You do not normally calculate these derivatives by hand. Autograd does it even when a model contains millions of trainable values.

## Example 3 — Repeat the process to learn a line

One update improves one prediction. Training repeats the same four moves.

The data below follows `y = 3x + 0.5`. We deliberately start with the wrong weight and bias, then let autograd guide both values.

```python
# A small dataset: y = 3x + 0.5
x = torch.linspace(-1, 1, 40).reshape(-1, 1)
y = 3 * x + 0.5

# Trainable values start with bad guesses
weight = torch.tensor([[0.0]], requires_grad=True)
bias = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.2

for step in range(40):
    # 1. Forward
    prediction = x @ weight + bias

    # 2. Loss: average squared error
    loss = (prediction - y).pow(2).mean()

    # Clear gradients left by the previous step
    weight.grad = None
    bias.grad = None

    # 3. Backward
    loss.backward()

    # 4. Update
    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    if step in (0, 1, 5, 20, 39):
        print(
            f"step {step:2d} | loss {loss.item():.6f} | "
            f"weight {weight.item():.3f} | bias {bias.item():.3f}"
        )

print(f"learned line: y = {weight.item():.3f}x + {bias.item():.3f}")
assert abs(weight.item() - 3.0) < 0.02
assert abs(bias.item() - 0.5) < 0.02
```

{{< output >}}
step  0 | loss 3.403846 | weight 0.421 | bias 0.200
step  1 | loss 2.421657 | weight 0.782 | bias 0.320
step  5 | loss 0.698077 | weight 1.788 | bias 0.477
step 20 | loss 0.007505 | weight 2.874 | bias 0.500
step 39 | loss 0.000024 | weight 2.993 | bias 0.500
learned line: y = 2.993x + 0.500
{{< /output >}}

## The four rules that prevent most autograd bugs

1. Put `requires_grad=True` on **trainable values**, not ordinary input data.
2. Reduce many errors to one scalar loss with `.mean()` or `.sum()` before calling `backward()`.
3. Clear old gradients before the next `backward()` call.
4. Update trainable values inside `torch.no_grad()` because the update is not part of the model's prediction.

The next example shows why rule 3 matters.

## Common bug — Gradients accumulate

PyTorch **adds** new gradients to `.grad`; it does not replace them. This is useful for some advanced techniques, but it is usually a bug in a basic training loop.

```python
x = torch.tensor(2.0, requires_grad=True)

# First fresh forward and backward
first_loss = x ** 2
first_loss.backward()
print("after first backward: ", x.grad.item())

# Second fresh forward and backward, but no clearing
second_loss = x ** 2
second_loss.backward()
print("after second backward:", x.grad.item(), "<- 4 + 4")

# Fix: clear before the next backward call
x.grad = None
third_loss = x ** 2
third_loss.backward()
print("after clearing first:  ", x.grad.item())

assert x.grad.item() == 4.0
```

{{< output >}}
after first backward:  4.0
after second backward: 8.0 <- 4 + 4
after clearing first:   4.0
{{< /output >}}

## `no_grad()` and `detach()` have different jobs

- Use `torch.no_grad()` around a **block of operations** that should not be recorded, such as validation or a manual weight update.
- Use `.detach()` when you need **one tensor value** without its computation history, such as logging a prediction.

```python
value = torch.tensor(3.0, requires_grad=True)
tracked_result = value * 2

with torch.no_grad():
    untracked_block_result = value * 2

detached_result = tracked_result.detach()

print("tracked result requires grad:  ", tracked_result.requires_grad)
print("no_grad result requires grad:  ", untracked_block_result.requires_grad)
print("detached result requires grad: ", detached_result.requires_grad)

assert tracked_result.requires_grad
assert not untracked_block_result.requires_grad
assert not detached_result.requires_grad
```

{{< output >}}
tracked result requires grad:   True
no_grad result requires grad:   False
detached result requires grad:  False
{{< /output >}}

## What this looks like in normal PyTorch code

Real models register their trainable values as parameters, and an optimizer performs the update. Autograd's job is still exactly the same: `loss.backward()` fills each parameter's `.grad`.

```python
import torch.nn as nn

torch.manual_seed(0)
x = torch.linspace(-1, 1, 40).reshape(-1, 1)
y = 3 * x + 0.5

model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(120):
    prediction = model(x)
    loss = (prediction - y).pow(2).mean()

    optimizer.zero_grad()  # clear every parameter's old gradient
    loss.backward()        # autograd fills every parameter's .grad
    optimizer.step()       # update every parameter

learned_weight = model.weight.item()
learned_bias = model.bias.item()

print(f"final loss:  {loss.item():.8f}")
print(f"learned line: y = {learned_weight:.3f}x + {learned_bias:.3f}")
assert loss.item() < 1e-4
```

{{< output >}}
final loss:  0.00000010
learned line: y = 3.000x + 0.500
{{< /output >}}

## Optional peek — What did PyTorch record?

A starting tensor created by you is called a **leaf**. Operations create new tensors with a `grad_fn` that points to the recorded backward operation. You rarely need to inspect this graph, but it is useful when debugging.

```python
a = torch.tensor(2.0, requires_grad=True)  # leaf
b = a * 3                                 # result of an operation
c = (b + 1) ** 2                          # scalar result

print("a.grad_fn:", a.grad_fn, "(leaf created by us)")
print("b.grad_fn:", type(b.grad_fn).__name__)
print("c.grad_fn:", type(c.grad_fn).__name__)

c.backward()
print("a.grad:", a.grad.item())
assert a.grad.item() == 42.0
```

{{< output >}}
a.grad_fn: None (leaf created by us)
b.grad_fn: MulBackward0
c.grad_fn: PowBackward0
a.grad: 42.0
{{< /output >}}

## Quick debugging guide

| Symptom | First thing to check |
|---|---|
| `.grad` is `None` | Did the trainable tensor use `requires_grad=True`, and did you call `backward()`? |
| Gradient grows every step | Clear it before each backward pass. |
| `backward()` complains about a second call | Recompute the forward pass to create a fresh graph. |
| Loss has many values | Reduce it with `.mean()` or `.sum()`. |
| Update becomes part of the graph | Put the manual update inside `torch.no_grad()`. |

## Practical checkpoint

Without copying the earlier loop, train one scalar `weight` so that `weight * 4` approaches `20`. Start with `weight = 0`, use `learning_rate = 0.02`, and run 20 steps. For every step:

1. make the prediction;
2. calculate squared error;
3. clear the old gradient;
4. call `backward()`;
5. update inside `torch.no_grad()`.

Done means the final prediction is within `0.01` of `20` and you can explain what the gradient told the weight.
