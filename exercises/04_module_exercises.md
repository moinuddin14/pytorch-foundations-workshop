# Module 4 — Module implementation lab

Build and inspect real modules. Runnable solutions: `python exercises/answers/04_module_answers.py`.

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
