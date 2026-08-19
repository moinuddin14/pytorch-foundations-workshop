# Module 5 — Training-loop implementation lab

Start from blank cells wherever possible. Runnable solutions: `python exercises/answers/05_training_loop_answers.py`.

## 1. Train a line from memory

Generate data for `y = 3*x + 0.5`, create `nn.Linear(1, 1)`, and write the five-step loop without copying it. Assert the learned weight and bias are within `0.02` of the targets.

## 2. Reproduce the missing-`zero_grad` bug

Train identical models with and without `zero_grad()`. Record both loss curves and assert the correct loop ends with lower loss.

## 3. Run a learning-rate sweep

Train with `lr` values `0.0001`, `0.01`, `0.5`, and `5.0`. Store final losses in a dictionary, print it, and identify slow, useful, and divergent rates from the results.

## 4. Compare batch sizes

Train 400 points with batch sizes `16` and `400` for ten epochs. Count optimizer steps and compare learned weights. Explain the difference using the measured step counts.

## 5. Add validation

Split synthetic data into training and validation sets. After every epoch:

```python
model.eval()
with torch.no_grad():
    val_loss = loss_fn(model(x_val), y_val)
model.train()
```

Store train and validation losses and assert validation does not change any parameter.

Done means the script trains, diagnoses a broken loop, compares hyperparameters, and validates safely.
