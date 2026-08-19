"""Runnable solutions for Module 2. Run: python exercises/answers/02_tensors_answers.py"""

import time

import torch


def shape_surgery():
    t = torch.arange(12)
    assert t.reshape(3, 4).shape == (3, 4)
    assert t.reshape(2, 2, 3).shape == (2, 2, 3)
    assert t.reshape(-1).shape == (12,)
    try:
        t.reshape(5, 2)
    except RuntimeError as error:
        print("reshape error:", str(error).splitlines()[0])


def broadcasting():
    result = torch.zeros(3, 1) + torch.zeros(4)
    assert result.shape == (3, 4)
    try:
        torch.zeros(3, 4) + torch.zeros(2, 4)
    except RuntimeError as error:
        print("broadcast error:", str(error).splitlines()[0])


def strides():
    t = torch.arange(6).reshape(2, 3)
    assert t.stride() == (3, 1)
    assert t.T.stride() == (1, 3)
    assert t.untyped_storage().data_ptr() == t.T.untyped_storage().data_ptr()
    try:
        t.T.view(6)
    except RuntimeError:
        pass
    assert t.T.reshape(6).tolist() == [0, 3, 1, 4, 2, 5]


def vectorization():
    x = torch.randn(100_000)
    start = time.perf_counter()
    loop_sum = sum(x[i].item() for i in range(x.numel()))
    loop_time = time.perf_counter() - start
    start = time.perf_counter()
    tensor_sum = x.sum().item()
    tensor_time = time.perf_counter() - start
    assert abs(loop_sum - tensor_sum) < 1e-3
    print(f"vectorized sum: {loop_time / tensor_time:.0f}x faster")


def dtype_and_device():
    labels = torch.tensor([0, 1, 0])
    predictions = torch.tensor([0.1, 0.9, 0.2])
    assert (predictions - labels.float()).dtype == torch.float32

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else None)
    if device:
        try:
            predictions.to(device) + labels
        except RuntimeError as error:
            print("device error:", str(error).splitlines()[0])


if __name__ == "__main__":
    shape_surgery()
    broadcasting()
    strides()
    vectorization()
    dtype_and_device()
    print("Module 2 solutions: PASS")
