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
