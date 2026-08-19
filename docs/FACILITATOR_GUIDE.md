# Practical-First Facilitator Runbook

**AI Yatra · PyTorch Foundations · Saturday, 22 August 2026 · 10:00 AM–1:30 PM IST**

## Delivery rule

Use **predict → run → modify → explain** for every example. Keep any uninterrupted explanation under three minutes. Target at least 70% learner activity and at most 30% explanation.

The only theory learners must retain is:

- tensors carry data and shape;
- autograd records operations and fills gradients;
- modules register parameters;
- training repeats forward, loss, zero gradients, backward, and step.

## Preflight

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__)"
jupyter notebook
```

- Run all seven notebooks once.
- Open Notebooks 01–05; keep 06–07 hidden until project reveals.
- Keep both project starters open in an editor.
- Keep solution files closed. They are recovery tools, not presentation material.

## Live schedule

| Clock | Segment | Learners spend most of the time… |
|---|---|---|
| 10:00–10:05 | Kickoff | predicting what the training loop will do |
| 10:05–10:25 | Notebook 01 | running NumPy/autograd and modifying a trained line |
| 10:25–10:55 | Notebook 02 | predicting shapes, breaking `view`, broadcasting, vectorizing |
| 10:55–11:05 | Break | — |
| 11:05–11:35 | Notebook 03 | calculating gradients, reproducing accumulation, checking derivatives |
| 11:35–11:55 | Notebook 04 | fixing parameter registration and testing save/eval behavior |
| 11:55–12:15 | Notebook 05 | writing the loop, reimplementing SGD, sweeping learning rates |
| 12:15–12:25 | Break | — |
| 12:25–1:00 | Project 1 | implementing and testing `MyLinear` |
| 1:00–1:25 | Project 2 | implementing primitives and training XOR |
| 1:25–1:30 | Wrap | rebuilding the loop from memory |

## 10:00–10:05 — Start with code

Show only this:

```python
pred = model(x)
loss = loss_fn(pred, y)
opt.zero_grad()
loss.backward()
opt.step()
```

Ask learners to mark the one line they understand least. Tell them they will implement or inspect every line today.

## 10:05–10:25 — Module 1

Open [Notebook 01](../notebooks/01_what_is_pytorch.ipynb).

1. Run the NumPy formula.
2. Before the PyTorch cell, ask for the gradient at `x=2`; run and verify.
3. Learners change `x` to `3` and rerun.
4. Run the complete training cell.
5. Learners change the target to `y = -2*x + 1` and verify the learned weight/bias.

Say only: “PyTorch adds a recorded computation and gradients to tensor operations.”

Success evidence: learners can change the synthetic line and predict the learned parameters.

## 10:25–10:55 — Module 2

Open [Notebook 02](../notebooks/02_tensors_properly.ipynb).

| Minutes | Learner task |
|---:|---|
| 5 | Predict indexing shapes; create a `(16, 3, 32, 32)` batch. |
| 5 | Flatten each image with `reshape(batch, -1)` and assert the shape. |
| 6 | Predict transpose strides; reproduce failing `view`; implement two fixes. |
| 5 | Predict three broadcast shapes; create and catch one incompatible case. |
| 4 | Move model and batch to the selected device and assert they match. |
| 5 | Time the element loop and replace it with `sum()`. |

Theory limit: define stride in one sentence and broadcasting in one sentence. Let the failing code provide the rest of the explanation.

Success evidence: each learner completes the practical checkpoint at the bottom of the notebook.

## 10:55–11:05 — Break

## 11:05–11:35 — Module 3

Open [Notebook 03](../notebooks/03_autograd_without_magic.ipynb).

1. Learners predict `dy/dx` and `dy/dw` before `backward()`.
2. They modify `b` so it receives a gradient and assert `b.grad == 1`.
3. Run one manual regression update; print weights before and after.
4. Reproduce gradient accumulation, then repair it.
5. Compare normal execution with `no_grad()`.
6. Implement and run the central-difference gradient check at three points.

Say only: “Backward walks recorded operations in reverse, applies the chain rule, and adds results to `.grad`.”

Do not skip accumulation or gradcheck.

Success evidence: learners implement two scalar gradient-descent steps without an optimizer.

## 11:35–11:55 — Module 4

Open [Notebook 04](../notebooks/04_nn_module_and_parameters.ipynb).

- Run `named_parameters()` on the example model.
- Learners repair the plain-tensor registration bug and assert the name appears.
- Save, restore, and compare every parameter.
- Run Dropout repeatedly in train and eval modes; assert which outputs match.
- If time remains, learners build a `4 -> 8 -> 2` module from a blank cell.

Say only: “A parameter assigned to a module is registered, discoverable, and trainable.”

Success evidence: the learner-built MLP passes output-shape, parameter-count, and save/reload assertions.

## 11:55–12:15 — Module 5

Open [Notebook 05](../notebooks/05_the_training_loop.ipynb).

1. Learners write the five-step loop from memory before revealing it.
2. Run the manual-SGD comparison and assert equality with `optimizer.step()`.
3. Learners add `lr=0.01` to the sweep and classify every result as slow, useful, or divergent.
4. Change batch size and count optimizer steps.
5. Add validation and confirm it does not change any parameter.

Say only: “For plain SGD, step subtracts learning-rate times gradient from every parameter.”

Success evidence: learners train `y = 4*x - 2` from a blank cell.

## 12:15–12:25 — Break

Ask everyone to open [Project 1](../projects/project1_linear_from_scratch.py).

## 12:25–1:00 — Project 1

1. Give 12 minutes for TODOs 1–3. Learners run after every TODO.
2. Use the starter's three checkpoints as the progress board:
   - parameters registered with correct shapes;
   - forward pass matches `nn.Linear`;
   - custom layer learns the target line.
3. Pair learners who pass Checkpoint 1 with anyone still debugging registration.
4. At minute 17, reveal [Notebook 06](../notebooks/06_project1_build_nn_linear.ipynb).
5. Run the numeric gradient check together.
6. Learners change the target to `y = -3*x + 0.25` and retrain.

If blocked, compare one TODO with [the solution](../projects/answers/project1_linear_from_scratch.py), not the whole file.

## 1:00–1:25 — Project 2

Open [Project 2](../projects/project2_mlp_from_scratch.py).

1. Give 7 minutes for ReLU, softmax, and cross-entropy. Run `check_primitives()`.
2. Give 6 minutes for `MyLinear` and `MyMLP`. Pass the shape/parameter checkpoint.
3. Train and watch loss at steps 0, 499, and 1999.
4. Confirm predictions `[0, 1, 1, 0]`.
5. Reveal [Notebook 07](../notebooks/07_project2_neural_network_from_scratch.ipynb) and its decision boundary.
6. Remove ReLU and retrain if time remains.

If blocked, use [the solution](../projects/answers/project2_mlp_from_scratch.py) one function at a time.

## 1:25–1:30 — Exit check

From a blank cell, ask learners to write the five-step loop. Then ask:

1. Where are weights? `nn.Parameter` objects registered in modules.
2. Where do gradients come from? `backward()` over the recorded graph.
3. What does plain SGD do? `parameter -= lr * parameter.grad`.

Share [the cheatsheet](CHEATSHEET.md) and the four implementation labs in `exercises/`.

## Recovery rules

| Problem | Action |
|---|---|
| Install/kernel failure | Pair the learner; use saved outputs; keep them editing a partner's code. |
| Ten minutes behind | Skip repeated demonstrations, not learner coding or project checks. |
| Starter crashes before a TODO is complete | Read the first traceback line and compare only that TODO with the solution. |
| Loss is `nan` | Lower learning rate, then check gradient clearing. |
| Accelerator issue | Use CPU; all examples are intentionally small. |

## Self-study loop

For each numbered notebook: predict, run, modify, complete its final checkpoint. Then complete the matching implementation lab and run its answer script only after attempting it. Finish both project starters without opening the reveals first.
