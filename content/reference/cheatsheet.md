+++
title = "PyTorch implementation cheatsheet"
description = "Shapes, devices, gradients, modules, loops, validation, and debugging in one compact page."
weight = 1
completable = false
+++
Keep this beside your editor. Copy a pattern, run it, then adapt it.

## The mental model

<figure class="article-figure">
  <a href="../../images/pytorch-training-mental-model.svg">
    <img src="../../images/pytorch-training-mental-model.svg" alt="Five-step PyTorch training loop: make a prediction, measure loss, clear old gradients, use autograd to calculate parameter gradients, and update weights before the next step.">
  </a>
  <figcaption>Read top to bottom. Blue is the forward pass, purple is the backward pass, and orange shows updated weights returning to the model. Tap the diagram to open the scalable version, or <a href="../../images/pytorch-training-mental-model.png">download the high-resolution PNG</a>.</figcaption>
</figure>

**PyTorch = tensors + autograd + `nn`.** Every feature serves this loop.

## The three flyer questions

| Question | Answer |
|---|---|
| Where are model weights stored? | Inside modules, as `nn.Parameter`s — serialized by `model.state_dict()` |
| Where do gradients come from? | `backward()` walks the recorded computation graph in reverse, applying the chain rule, accumulating into `.grad` |
| What happens during `optimizer.step()`? | `param -= lr * param.grad` (SGD); every optimizer is a fancier version of that loop |

## The training loop (write this from memory)

```python
model = model.to(device)
model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)              # 1. forward
        loss = loss_fn(pred, yb)      # 2. loss
        opt.zero_grad()               # 3. reset gradients (they accumulate!)
        loss.backward()               # 4. backward — fill .grad
        opt.step()                    # 5. update weights
```

## Tensor essentials

| Task | Snippet |
|---|---|
| Create | `torch.zeros(2, 3)`, `torch.randn(2, 3)`, `torch.tensor([[1., 2.]])` |
| Shape | `t.shape`, `t.reshape(-1)`, `t.view(2, 3)` (view needs contiguity), `t.T` |
| Device | `t.to(device)`; choose CUDA, then Apple MPS, then CPU (see below) |
| dtype | `t.float()`, `t.long()`; check with `t.dtype` |
| Read one value | `t.item()` (scalar), `t.tolist()` (list) |
| Stop tracking | `with torch.no_grad():` (sections), `t.detach()` (one tensor) |
| Reproducibility | `torch.manual_seed(0)` |
| In-place ops | trailing underscore: `x.zero_()`, `x.add_(1)` |

**Broadcasting:** align shapes from the right; dims match, are `1`, or are missing. `(4,1) + (1,5) → (4,5)`, no copy.

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
model = model.to(device)
x = x.to(device)
```

## Autograd essentials

| Task | Snippet |
|---|---|
| Track a tensor | `x = torch.tensor(2.0, requires_grad=True)` |
| Inspect the graph | `y.grad_fn` (leaves have `None`) |
| Compute gradients | `loss.backward()` → results in `x.grad` |
| Reset gradients | `opt.zero_grad()` or `x.grad.zero_()` |
| Verify gradients | central difference: `(f(x+ε) − f(x−ε)) / 2ε` with `ε = 1e-3` in float32 |

## nn essentials

| Task | Snippet |
|---|---|
| Make a weight trainable | `self.w = nn.Parameter(torch.randn(...))` — plain tensors on `self` are invisible |
| List weights | `model.named_parameters()` |
| Save / load | `torch.save(model.state_dict(), p)` / `model.load_state_dict(torch.load(p, weights_only=True))` |
| Train vs eval | `model.train()` / `model.eval()` — dropout & batchnorm behave differently |
| Validate | `model.eval()` **and** `with torch.no_grad():` — different mechanisms, use both |
| Call a model | `model(x)` (never `model.forward(x)`) |

## The error decoder

| Error | Cause | Fix |
|---|---|---|
| `Expected all tensors to be on the same device` | model and data on different devices | `model.to(device)` and `x.to(device)` |
| `The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0` | shapes not broadcastable | fix the shapes (align from the right) |
| `view size is not compatible with input tensor's size and stride` | `view()` on a non-contiguous tensor (e.g. after `.T`) | use `reshape()`, or `.contiguous().view()` |
| `shape '[5, 2]' is invalid for input of size 12` | reshape element count mismatch | `prod(new_shape) == numel` |
| `RuntimeError: Trying to backward through the graph a second time` | calling `backward()` twice on the same graph | re-run the forward pass (or `retain_graph=True`) |
| `optimizer got an empty parameter list` | no registered parameters (plain tensors instead of `nn.Parameter`) | wrap weights in `nn.Parameter` |
| `Error(s) in loading state_dict ... Missing key(s)` | loading into a different architecture | save/load the same architecture's `state_dict` |
| Loss becomes `nan` | learning rate too high (usually) | lower `lr` by 10×; check `zero_grad()` is called |
| Loss doesn't decrease | lr too small, or gradients not flowing | sweep `lr`; verify `backward()` runs and `.grad` is populated |

## The five habits

1. Call `opt.zero_grad()` before every `loss.backward()`.
2. Never loop over tensor elements in Python — use vectorized ops.
3. `model.eval()` + `torch.no_grad()` for validation; `model.train()` to go back.
4. Save `state_dict()`, not the model object.
5. When something breaks, print shapes — `print(x.shape)` solves half of all bugs.

## Add checks while building

```python
assert output.shape == expected_shape
assert torch.isfinite(loss)
assert all(p.grad is not None for p in model.parameters())
assert next(model.parameters()).device == x.device
```

Replace silent assumptions with assertions; remove only those that become too expensive at production scale.
