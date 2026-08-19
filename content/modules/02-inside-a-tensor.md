+++
title = "Inside a tensor"
description = "Work with shapes, strides, broadcasting, dtypes, devices, and vectorized operations."
weight = 2
completable = true
notebook = "notebooks/02_tensors_properly.ipynb"
+++
## Lab 1: read and change shapes

A tensor is an array with a `shape`, `ndim`, `dtype`, and `device`.

```python
import torch

scalar = torch.tensor(5.0)                      # 0-d
vector = torch.tensor([1.0, 2.0, 3.0])          # 1-d
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) # 2-d
cube   = torch.zeros(2, 3, 4)                   # 3-d

for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix), ("cube", cube)]:
    print(f"{name:7s} shape={str(t.shape):10s} ndim={t.ndim} numel={t.numel()}")
```

{{< output >}}
scalar  shape=torch.Size([]) ndim=0 numel=1
vector  shape=torch.Size([3]) ndim=1 numel=3
matrix  shape=torch.Size([2, 2]) ndim=2 numel=4
cube    shape=torch.Size([2, 3, 4]) ndim=3 numel=24
{{< /output >}}

What do you think are the shapes of `t[1]`, `t[1, 2]`, and `t[:, 0, :]`?

```python
t = torch.arange(24).reshape(2, 3, 4)   # 0..23 as 2 tables of 3×4
print("t =", t, sep="\n")
print("\nt[1]        (second table) =", t[1].shape, t[1].flatten().tolist())
print("t[1, 2]     (third row of it) =", t[1, 2].tolist())
print("t[1, 2, 3]  (single element)  =", t[1, 2, 3].item())
print("t[:, 0, :]  (first row of each table) =", t[:, 0, :].shape)
```

{{< output >}}
t =
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],

        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])

t[1]        (second table) = torch.Size([3, 4]) [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
t[1, 2]     (third row of it) = [20, 21, 22, 23]
t[1, 2, 3]  (single element)  = 23
t[:, 0, :]  (first row of each table) = torch.Size([2, 4])
{{< /output >}}

### Mini-build: prepare images for a linear layer

Use `reshape(batch_size, -1)` to turn `(8, 3, 64, 64)` into `(8, 12288)`. `-1` asks PyTorch to infer that dimension. Use `permute` only when you need to reorder dimensions.

```python
batch = torch.randn(8, 3, 64, 64)     # 8 images, 3 channels, 64x64
flat = batch.reshape(8, -1)           # flatten each image to a vector
print("batch shape:", batch.shape)
print("flattened shape:", flat.shape, "  (3*64*64 =", 3*64*64, ")")

# reshape(-1) flattens everything:
print("fully flat:", batch.reshape(-1).shape)
assert flat.shape == (8, 3 * 64 * 64)
```

{{< output >}}
batch shape: torch.Size([8, 3, 64, 64])
flattened shape: torch.Size([8, 12288])   (3*64*64 = 12288 )
fully flat: torch.Size([98304])
{{< /output >}}

## Lab 2: inspect storage and strides

A tensor combines stored numbers with metadata describing how to read them. `stride` is the storage jump for one step along each dimension. Run the cell and calculate the storage position of `[1, 1]` yourself.

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("shape :", t.shape)
print("stride:", t.stride())   # (3, 1): move 3 in storage to go down a row, 1 to go right
print("storage order (the numbers in memory):", t.flatten().tolist())
```

{{< output >}}
shape : torch.Size([2, 3])
stride: (3, 1)
storage order (the numbers in memory): [1, 2, 3, 4, 5, 6]
{{< /output >}}

Predict the shape and stride of `t.T`, then run the cell. Confirm that `t` and `t.T` share storage.

```python
t = torch.arange(6).reshape(2, 3)
tt = t.T                      # transpose
print("t :", t.shape, "stride", t.stride())
print("tT:", tt.shape, "stride", tt.stride())

# Both point at the SAME storage:
same_storage = t.untyped_storage().data_ptr() == tt.untyped_storage().data_ptr()
print("\nsame underlying data?", same_storage)
assert tt.shape == (3, 2) and tt.stride() == (1, 3) and same_storage

# Transpose is O(1) — it only swapped the stride tuple (3,1) -> (1,3).
```

{{< output >}}
t : torch.Size([2, 3]) stride (3, 1)
tT: torch.Size([3, 2]) stride (1, 3)

same underlying data? True
{{< /output >}}

### Break it, then fix it

A transpose is often non-contiguous, so `view()` can fail. Run the failing call, then fix it twice: with `reshape()` and with `contiguous().view()`.

```python
t = torch.arange(6).reshape(2, 3)
tt = t.T
print("is t.T contiguous?", tt.is_contiguous())
try:
    tt.view(6)
except RuntimeError as e:
    print("view(6) failed:", str(e)[:80], "...")

print("reshape(6) works:", tt.reshape(6).tolist())
print("contiguous() then view works:", tt.contiguous().view(6).tolist())
```

{{< output >}}
is t.T contiguous? False
view(6) failed: view size is not compatible with input tensor's size and stride (at least one di ...
reshape(6) works: [0, 3, 1, 4, 2, 5]
contiguous() then view works: [0, 3, 1, 4, 2, 5]
{{< /output >}}

## Lab 3: broadcasting

Align shapes from the right. Each pair of dimensions must match or one must be `1`. Predict every result shape before running the next two cells, then invent one incompatible pair.

```python
# Example: add a row-vector to every row of a matrix
m = torch.arange(6).reshape(2, 3).float()   # (2, 3)
r = torch.tensor([10.0, 20.0, 30.0])        # (3,)
print(m + r)

# Example: scale every element by a scalar
print("\n", m * 2.0)

# Shapes align from the right: (2,3) vs (3,)  →  (2,3) vs (1,3)  →  broadcast to (2,3)
```

{{< output >}}
tensor([[10., 21., 32.],
        [13., 24., 35.]])

 tensor([[ 0.,  2.,  4.],
        [ 6.,  8., 10.]])
{{< /output >}}

```python
a = torch.zeros(4, 1)      # (4, 1)
b = torch.zeros(1, 5)      # (1, 5)
c = a + b
print("a (4,1) + b (1,5) ->", c.shape)   # both stretch: (4, 5)
assert c.shape == (4, 5)

try:
    torch.zeros(4, 3) + torch.zeros(2, 3)
except RuntimeError as e:
    print("\n(4,3) + (2,3) fails:", e)
```

{{< output >}}
a (4,1) + b (1,5) -> torch.Size([4, 5])

(4,3) + (2,3) fails: The size of tensor a (4) must match the size of tensor b (2) at non-singleton dimension 0
{{< /output >}}

### Check memory behavior

Run `expand` and compare storage pointers. The expanded view reuses the original storage; the arithmetic result still allocates its output.

```python
big = torch.randn(1000, 1000)
row = torch.randn(1000)

# Method 1: explicit expansion (copies?)
expanded = row.expand(1000, 1000)   # note: expand() is also a view — no copy!
print("expand shares storage:", expanded.untyped_storage().data_ptr() == row.untyped_storage().data_ptr())

# Method 2: broadcast add
out = big + row
print("broadcast result shape:", out.shape)
assert expanded.untyped_storage().data_ptr() == row.untyped_storage().data_ptr()
assert out.shape == (1000, 1000)
print("correct?", torch.allclose(out, big + expanded))
```

{{< output >}}
expand shares storage: True
broadcast result shape: torch.Size([1000, 1000])
correct? True
{{< /output >}}

## Lab 4: dtype and device

Models usually use `float32`; class indices usually use `int64`. Every tensor in one operation must be on the same device. Run the cell, then explicitly convert an integer tensor with `.float()`.

```python
x = torch.tensor([1, 2, 3])            # int64 by default
print("int tensor:", x.dtype)
print("as float:", x.float().dtype)

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print("device in use:", device)

t = torch.randn(2, 3, device=device)
print("tensor on", t.device)
print("moved back to cpu:", t.cpu().device)
```

{{< output >}}
int tensor: torch.int64
as float: torch.float32
device in use: mps
tensor on mps:0
moved back to cpu: cpu
{{< /output >}}

### Implement the device pattern

Move both the model and every input batch to the selected device:

```python
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
model = model.to(device)     # move the model's parameters
x = x.to(device)
```

Print `next(model.parameters()).device` and `x.device` before the forward pass.

```python
import torch.nn as nn
# The canonical pattern, live:
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
model_demo = nn.Linear(4, 2).to(device)      # model -> device
x_demo = torch.randn(3, 4).to(device)        # data  -> device
print("model weights on:", next(model_demo.parameters()).device)
print("input on        :", x_demo.device)
output_demo = model_demo(x_demo)
print("forward pass works:", output_demo.shape)
assert next(model_demo.parameters()).device == x_demo.device
assert output_demo.shape == (3, 2)
```

{{< output >}}
model weights on: mps:0
input on        : mps:0
forward pass works: torch.Size([3, 2])
{{< /output >}}

## Lab 5: replace a Python loop

Time element-by-element summation against `x.sum()`. Then replace the loop in any one of your own examples with a tensor operation.

```python
import time

n = 1_000_000
x = torch.randn(n)

# Slow way: Python loop
start = time.perf_counter()
s = 0.0
for i in range(n):
    s += x[i].item()
slow = time.perf_counter() - start

# Fast way: vectorized
start = time.perf_counter()
s2 = x.sum().item()
fast = time.perf_counter() - start

print(f"Python loop: {slow*1000:.1f} ms")
print(f"Vectorized : {fast*1000:.2f} ms")
print(f"Speedup    : {slow/fast:.0f}x")
same_answer = abs(s - s2) < 1e-3
print("same answer?", same_answer)
assert same_answer
```

{{< output >}}
Python loop: 511.2 ms
Vectorized : 0.20 ms
Speedup    : 2511x
same answer? True
{{< /output >}}

## Practical checkpoint

Create a `(4, 3, 32, 32)` image batch, flatten each image, normalize each feature with broadcasting, move it to `device`, and compute its mean without an element loop. Print the shape after every step.
