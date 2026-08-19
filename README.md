# PyTorch Foundations — Tensor → Gradient → Model

**AI Yatra · Session 1 · Saturday, 22 August 2026 · 10:00 AM–1:30 PM IST**

This is a 3.5-hour implementation workshop for learners with little or no PyTorch experience. Roughly 70% of the live time is spent predicting, running, changing, debugging, or writing code.

## Workshop website

**[Open the phone-friendly workshop →](https://moinuddin14.github.io/pytorch-foundations-workshop/)**

The Hugo site contains the complete learning path, compact explanations, runnable examples, four implementation labs, two projects, progress tracking, and one-tap Google Colab links. Participants can read and complete the workshop from a phone without installing Python.

## Start here

- **Facilitating the workshop:** follow [the facilitator runbook](docs/FACILITATOR_GUIDE.md). It is the single source of truth for timing, file order, prompts, transitions, breaks, and fallback options.
- **Attending the workshop:** open the [workshop site](https://moinuddin14.github.io/pytorch-foundations-workshop/). In every lab: predict first, run second, modify third, explain last.
- **Practising afterwards:** complete the matching implementation lab after Modules 2–5, then both project starters. Every answer is a runnable `.py` script with assertions.
- **Need a quick reminder:** use the [one-page cheatsheet](docs/CHEATSHEET.md).

## The one workshop path

| Order | Live material | What you implement or test |
|---:|---|---|
| 1 | [What exactly is PyTorch?](notebooks/01_what_is_pytorch.ipynb) | Compare NumPy with autograd; modify a complete training loop. |
| 2 | [Inside a tensor](notebooks/02_tensors_properly.ipynb) | Reshape image batches, debug contiguity, broadcast, move devices, vectorize. |
| 3 | [Autograd: how PyTorch learns](notebooks/03_autograd_without_magic.ipynb) | Improve one prediction with gradients, train a line manually, and fix accumulation. |
| 4 | [`nn.Module` and parameters](notebooks/04_nn_module_and_parameters.ipynb) | Fix registration, save/reload state, test train/eval behavior. |
| 5 | [The training loop](notebooks/05_the_training_loop.ipynb) | Reimplement SGD, sweep learning rates, batch data, validate safely. |
| 6 | [Project 1 starter](projects/project1_linear_from_scratch.py), then [checks and reveal](notebooks/06_project1_build_nn_linear.ipynb) | Build, compare, gradient-check, and train `MyLinear`. |
| 7 | [Project 2 starter](projects/project2_mlp_from_scratch.py), then [checks and reveal](notebooks/07_project2_neural_network_from_scratch.ipynb) | Implement activation/loss functions and train an MLP on XOR. |
| 8 | [Cheatsheet](docs/CHEATSHEET.md) | Rebuild the five-step loop from memory. |

The exercises and answer files are intentionally outside the live path. They are practice and recovery material, not additional workshop modules.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook
```

Verify the environment before the workshop:

```bash
python -c "import torch; print(torch.__version__)"
```

Everything runs on CPU. CUDA and Apple Silicon MPS are used automatically when available in the device example, but no accelerator is required.

## Repository map

```text
PyTorch-Foundations-Workshop/
├── notebooks/                  # the seven-module live teaching sequence
├── projects/
│   ├── project1_...py          # participant starters
│   ├── project2_...py
│   └── answers/                # facilitator/participant solutions
├── exercises/
│   ├── 02_...md through 05_...md
│   └── answers/                # runnable solution scripts with assertions
├── docs/
│   ├── FACILITATOR_GUIDE.md    # the live runbook and self-study route
│   └── CHEATSHEET.md           # compact reference and error decoder
├── content/                     # Hugo learning pages generated from source material
├── layouts/                     # custom mobile-first Hugo templates
├── assets/                      # site CSS and JavaScript
├── static/                      # PWA manifest, icon, and service worker
├── scripts/build_site_content.py
├── hugo.toml
├── requirements.txt
└── README.md
```

## Build the site locally

Install Hugo 0.165.0 or newer, then run:

```bash
python3 scripts/build_site_content.py
hugo server
```

Pushing `main` deploys the site to GitHub Pages through `.github/workflows/hugo.yaml`.

## The three anchor questions

By the end, every participant should be able to answer these without notes:

1. **Where are model weights stored?** As `nn.Parameter` objects registered inside modules; `state_dict()` provides their named, serializable state.
2. **Where do gradients come from?** `backward()` walks the recorded computation graph in reverse and applies the chain rule, accumulating results in `.grad`.
3. **What does `optimizer.step()` do?** For SGD, it applies `parameter -= learning_rate * parameter.grad`; other optimizers compute a more sophisticated update from the same gradients.

That is the workshop: **Tensor → Gradient → Model**.
