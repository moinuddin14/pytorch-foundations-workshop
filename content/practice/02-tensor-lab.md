+++
title = "Tensor implementation lab"
description = "Five shape, broadcasting, storage, speed, dtype, and device challenges."
weight = 1
completable = true
notebook = "notebooks/labs/02_tensor_lab.ipynb"
+++
Create one notebook or Python file and complete all five tasks. Predict each shape before running the code.
## 1. Prepare a batch

Start with `t = torch.arange(12)`.

- Reshape it to `(3, 4)` and `(2, 2, 3)`.
- Flatten it with `reshape(-1)`.
- Catch the error from `reshape(5, 2)`.
- Add shape assertions after every operation.

## 2. Implement broadcasting examples

For `a = torch.zeros(3, 1)` and `b = torch.zeros(4)`:

- Predict and assert the shape of `a + b`.
- Add a `(3,)` bias to every row of a `(5, 3)` batch.
- Construct one incompatible pair and catch its `RuntimeError`.

## 3. Debug a non-contiguous tensor

Use `t = torch.arange(6).reshape(2, 3)`.

- Print the strides of `t` and `t.T`.
- Assert that they share storage.
- Reproduce the failing `t.T.view(6)` call.
- Fix it with both `reshape(6)` and `contiguous().view(6)`.

## 4. Remove an element loop

Create `x = torch.randn(100_000)`. Time a Python element loop and `x.sum()`. Assert that the results match within `1e-3`, then print the speedup.

## 5. Make dtype and device explicit

- Subtract integer labels from float predictions and inspect PyTorch's promoted dtype.
- Repeat with `labels.float()` so the conversion is explicit.
- Select CUDA, MPS, or CPU; move a model and a batch to that device; run a forward pass.
- Print and assert the model and batch devices match.

Done means the file runs from top to bottom with all assertions passing.


{{< details summary="Reveal the runnable solution" >}}

```python
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
```

{{< /details >}}
