+++
title = "Module implementation lab"
description = "Fix registration, restore state, test modes, and build a nested MLP."
weight = 3
completable = true
notebook = "notebooks/labs/04_module_lab.ipynb"
+++
Build and inspect real modules.
## 1. Fix parameter registration

Create a module containing an `nn.Parameter`, a plain tensor, and an `nn.Linear` submodule. Print `named_parameters()`, convert the plain tensor into a parameter, and assert its name appears.

## 2. Save and restore a model

Build `nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))`.

- Print every state-dict key and shape.
- Save, restore into a fresh model, and assert all parameters match.
- Attempt loading into the wrong architecture and catch the error.

## 3. Reproduce train/eval behavior

Run a model with `Dropout(p=0.9)` three times in train mode and three times in eval mode. Assert train outputs vary and eval outputs match.

## 4. Build a nested MLP

Implement a `4 -> 8 -> 8 -> 2` class with three linear submodules and ReLU activations. Pass a `(5, 4)` batch and assert the result is `(5, 2)`. Assert that all six parameter tensors are registered.

## 5. Inspect the optimizer's references

Compare parameter object IDs held by the model and optimizer. Assert the sets match, run one update, and assert at least one parameter changed.

Done means the module passes shape, registration, persistence, and update checks.


{{< details summary="Reveal the runnable solution" >}}

```python
"""Runnable solutions for Module 4. Run: python exercises/answers/04_module_answers.py"""

from tempfile import TemporaryDirectory
from pathlib import Path

import torch
import torch.nn as nn


class Weird(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(3))
        self.b = torch.randn(3)
        self.c = nn.Linear(3, 3)


class MLP3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def parameter_registration():
    model = Weird()
    names = [name for name, _ in model.named_parameters()]
    assert names == ["a", "c.weight", "c.bias"]
    assert not model.b.requires_grad


def state_dict_round_trip():
    model = MLP3()
    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = MLP3()
        restored.load_state_dict(torch.load(path, weights_only=True))
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters()))


def train_eval_modes():
    model = nn.Sequential(nn.Linear(4, 8), nn.Dropout(p=0.9), nn.Linear(8, 2))
    x = torch.ones(4, 4)
    model.train()
    train_outputs = [model(x) for _ in range(3)]
    model.eval()
    eval_outputs = [model(x) for _ in range(3)]
    assert not all(torch.equal(output, train_outputs[0]) for output in train_outputs)
    assert all(torch.equal(output, eval_outputs[0]) for output in eval_outputs)


def optimizer_references():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert model_ids == optimizer_ids


if __name__ == "__main__":
    parameter_registration()
    state_dict_round_trip()
    train_eval_modes()
    optimizer_references()
    print("Module 4 solutions: PASS")
```

{{< /details >}}
