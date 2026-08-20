# Module 2 — Tensor implementation lab

Create one notebook or Python file and complete all six tasks. Predict each shape before running the code. Runnable solutions: `python exercises/answers/02_tensors_answers.py`.

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

## 3. Compare element-wise and matrix multiplication

Create these two tensors:

```python
left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
```

- Predict and calculate both `left * right` and `left @ right`.
- Assert the complete expected value and shape of each result.
- Write out the row-by-column calculation that produces `(left @ right)[0, 0]`.
- Create `features` with shape `(4, 3)` and `weights` with shape `(3, 2)`. Assert the shape of `features @ weights`, then try `features * weights` and catch its `RuntimeError`.
- In one sentence, explain when to use `*` and when to use `@`.

## 4. Debug a non-contiguous tensor

Use `t = torch.arange(6).reshape(2, 3)`.

- Print the strides of `t` and `t.T`.
- Assert that they share storage.
- Reproduce the failing `t.T.view(6)` call.
- Fix it with both `reshape(6)` and `contiguous().view(6)`.

## 5. Remove an element loop

Create `x = torch.randn(100_000)`. Time a Python element loop and `x.sum()`. Assert that the results match within `1e-3`, then print the speedup.

## 6. Make dtype and device explicit

- Subtract integer labels from float predictions and inspect PyTorch's promoted dtype.
- Repeat with `labels.float()` so the conversion is explicit.
- Select CUDA, MPS, or CPU; move a model and a batch to that device; run a forward pass.
- Print and assert the model and batch devices match.

Done means the file runs from top to bottom with all assertions passing.
