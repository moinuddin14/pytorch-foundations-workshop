+++
title = "Modules and parameters"
description = "Register parameters correctly, save state, and understand training and evaluation modes."
weight = 4
completable = true
notebook = "notebooks/04_nn_module_and_parameters.ipynb"
+++
## Lab 1: build and inspect a module

Define layers in `__init__` and computation in `forward`. Run the cell, call `model(x)`, and inspect every item returned by `named_parameters()`.

```python
import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)   # a submodule
        self.fc2 = nn.Linear(8, 1)   # a submodule

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = TinyModel()
print(model)
print("\nparameters():")
for name, p in model.named_parameters():
    print(f"  {name:12s} shape={tuple(p.shape)}")
assert len(list(model.parameters())) == 4
assert model(torch.randn(5, 4)).shape == (5, 1)
```

{{< output >}}
TinyModel(
  (fc1): Linear(in_features=4, out_features=8, bias=True)
  (fc2): Linear(in_features=8, out_features=1, bias=True)
)

parameters():
  fc1.weight   shape=(8, 4)
  fc1.bias     shape=(8,)
  fc2.weight   shape=(1, 8)
  fc2.bias     shape=(1,)
{{< /output >}}

`nn.Linear` contains registered parameters, and nested modules are discovered recursively. Next, run the broken example and compare a plain tensor with `nn.Parameter`.

```python
class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_plain = torch.randn(3, 3)                 # plain tensor
        self.w_param = nn.Parameter(torch.randn(3, 3))   # parameter

b = Broken()
print("registered params:", [n for n, _ in b.named_parameters()])
print("w_plain requires_grad:", b.w_plain.requires_grad)
print("w_param requires_grad:", b.w_param.requires_grad)
assert [name for name, _ in b.named_parameters()] == ["w_param"]
```

{{< output >}}
registered params: ['w_param']
w_plain requires_grad: False
w_param requires_grad: True
{{< /output >}}

Fix `Broken` by converting `w_plain` to `nn.Parameter`, then assert that both weights appear in `dict(model.named_parameters())`.

## Lab 2: save and restore weights

`parameters()` supplies live trainable tensors to an optimizer. `state_dict()` supplies named state for saving. Run the round trip and verify every restored tensor with `torch.equal`.

```python
import os
import warnings
warnings.filterwarnings("ignore", message="TypedStorage")   # silence a benign torch 2.1 warning

sd = model.state_dict()
print("state_dict keys:")
for k, v in sd.items():
    print(f"  {k:12s} shape={tuple(v.shape)}")

# saving & loading — the canonical pattern:
path = "tiny_model.pt"
torch.save(model.state_dict(), path)

model2 = TinyModel()                 # fresh random weights
model2.load_state_dict(torch.load(path, weights_only=True))
weights_match = all(torch.equal(a, b) for a, b in zip(model.parameters(), model2.parameters()))
print("\nloaded weights match:", weights_match)
assert weights_match

os.remove(path)   # clean up the file — the check already passed
```

{{< output >}}
state_dict keys:
  fc1.weight   shape=(8, 4)
  fc1.bias     shape=(8,)
  fc2.weight   shape=(1, 8)
  fc2.bias     shape=(1,)

loaded weights match: True
{{< /output >}}

Change one restored weight inside `torch.no_grad()`, confirm equality fails, then load the state dict again and confirm equality returns.

## Lab 3: reproduce the train/eval difference

Run the same Dropout model three times in train mode and three times in eval mode. Compare the outputs.

```python
m = nn.Sequential(nn.Linear(4, 8), nn.Dropout(p=0.5), nn.Linear(8, 2))
x = torch.ones(4, 4)

m.train()
train_outputs = [m(x) for _ in range(3)]
train_varies = not all(torch.equal(out, train_outputs[0]) for out in train_outputs)
print("train mode varies:", train_varies)

m.eval()
eval_outputs = [m(x) for _ in range(3)]
eval_matches = all(torch.equal(out, eval_outputs[0]) for out in eval_outputs)
print("eval mode matches:", eval_matches)
assert train_varies and eval_matches and not m.training
```

{{< output >}}
train mode varies: True
eval mode matches: True
{{< /output >}}

Use this validation pattern, then switch training back on:

```python
model.eval()
with torch.no_grad():
    # validation loop here
model.train()   # back to training
```

## Implementation rule

Let the optimizer change weights during training. For deliberate initialization, use `with torch.no_grad(): model.weight.copy_(source)`. Do not use `.data`.

## Practical checkpoint

Build a `4 -> 8 -> 2` MLP class, run a `(5, 4)` batch through it, print all parameter names and shapes, and save/reload its state.
