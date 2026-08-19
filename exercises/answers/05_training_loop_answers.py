"""Runnable solutions for Module 5. Run: python exercises/answers/05_training_loop_answers.py"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_data(points=100):
    x = torch.linspace(-1, 1, points).reshape(-1, 1)
    return x, 3 * x + 0.5


def train(lr=0.1, steps=200, zero_grad=True):
    torch.manual_seed(0)
    x, y = make_data()
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loss = None
    for _ in range(steps):
        if zero_grad:
            optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model, loss.item()


def zero_grad_comparison():
    correct, correct_loss = train(zero_grad=True)
    broken, broken_loss = train(zero_grad=False)
    assert abs(correct.weight.item() - 3.0) < 1e-3
    assert correct_loss < broken_loss


def learning_rate_sweep():
    results = {lr: train(lr=lr)[1] for lr in (0.0001, 0.01, 0.5, 5.0)}
    print("learning-rate losses:", results)
    assert results[0.5] < results[0.0001]
    assert not torch.isfinite(torch.tensor(results[5.0]))


def batched_training(batch_size, epochs=10):
    x, y = make_data(points=400)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model


def evaluate():
    model, _ = train()
    x, y = make_data()
    model.eval()
    with torch.no_grad():
        loss = nn.MSELoss()(model(x), y)
    assert loss.item() < 1e-8


if __name__ == "__main__":
    zero_grad_comparison()
    learning_rate_sweep()
    small_batch = batched_training(16)
    full_batch = batched_training(400)
    print("batch=16 weight:", small_batch.weight.item())
    print("batch=400 weight:", full_batch.weight.item())
    evaluate()
    print("Module 5 solutions: PASS")
