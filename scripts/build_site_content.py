#!/usr/bin/env python3
"""Build the Hugo workshop pages and phone-friendly Colab lab notebooks."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "content"
LABS = ROOT / "notebooks" / "labs"

MODULES = [
    {
        "source": "notebooks/01_what_is_pytorch.ipynb",
        "target": "content/modules/01-what-is-pytorch.md",
        "title": "What PyTorch actually does",
        "description": "Run one formula, inspect its gradient, then train a tiny model end to end.",
        "weight": 1,
    },
    {
        "source": "notebooks/02_tensors_properly.ipynb",
        "target": "content/modules/02-inside-a-tensor.md",
        "title": "Inside a tensor",
        "description": "Work with shapes, strides, broadcasting, dtypes, devices, and vectorized operations.",
        "weight": 2,
    },
    {
        "source": "notebooks/03_autograd_without_magic.ipynb",
        "target": "content/modules/03-autograd-without-magic.md",
        "title": "Autograd: how PyTorch learns",
        "description": "See why gradients exist, use them to improve a prediction, and train a line from scratch.",
        "weight": 3,
    },
    {
        "source": "notebooks/04_nn_module_and_parameters.ipynb",
        "target": "content/modules/04-modules-and-parameters.md",
        "title": "Modules and parameters",
        "description": "Register parameters correctly, save state, and understand training and evaluation modes.",
        "weight": 4,
    },
    {
        "source": "notebooks/05_the_training_loop.ipynb",
        "target": "content/modules/05-training-loop.md",
        "title": "The training loop",
        "description": "Write SGD from memory, compare learning rates, batch data, and validate a model.",
        "weight": 5,
    },
]

PROJECTS = [
    {
        "source": "notebooks/06_project1_build_nn_linear.ipynb",
        "target": "content/projects/project-1-linear.md",
        "title": "Project 1: Rebuild nn.Linear",
        "description": "Implement a registered linear layer, compare it with PyTorch, check gradients, and train it.",
        "weight": 1,
        "notebook": "notebooks/labs/project1_linear_lab.ipynb",
        "starter": "projects/project1_linear_from_scratch.py",
        "answer": "projects/answers/project1_linear_from_scratch.py",
    },
    {
        "source": "notebooks/07_project2_neural_network_from_scratch.ipynb",
        "target": "content/projects/project-2-xor.md",
        "title": "Project 2: Solve XOR with your MLP",
        "description": "Implement activations and loss, compose a two-layer network, and prove why non-linearity matters.",
        "weight": 2,
        "notebook": "notebooks/labs/project2_xor_lab.ipynb",
        "starter": "projects/project2_mlp_from_scratch.py",
        "answer": "projects/answers/project2_mlp_from_scratch.py",
    },
]

PRACTICE = [
    {
        "source": "exercises/02_tensor_exercises.md",
        "answer": "exercises/answers/02_tensors_answers.py",
        "target": "content/practice/02-tensor-lab.md",
        "title": "Tensor implementation lab",
        "description": "Five shape, broadcasting, storage, speed, dtype, and device challenges.",
        "weight": 1,
        "notebook": "notebooks/labs/02_tensor_lab.ipynb",
    },
    {
        "source": "exercises/03_autograd_exercises.md",
        "answer": "exercises/answers/03_autograd_answers.py",
        "target": "content/practice/03-autograd-lab.md",
        "title": "Autograd implementation lab",
        "description": "Predict, inspect, break, fix, and numerically verify gradients.",
        "weight": 2,
        "notebook": "notebooks/labs/03_autograd_lab.ipynb",
    },
    {
        "source": "exercises/04_module_exercises.md",
        "answer": "exercises/answers/04_module_answers.py",
        "target": "content/practice/04-module-lab.md",
        "title": "Module implementation lab",
        "description": "Fix registration, restore state, test modes, and build a nested MLP.",
        "weight": 3,
        "notebook": "notebooks/labs/04_module_lab.ipynb",
    },
    {
        "source": "exercises/05_training_loop_exercises.md",
        "answer": "exercises/answers/05_training_loop_answers.py",
        "target": "content/practice/05-training-loop-lab.md",
        "title": "Training-loop implementation lab",
        "description": "Train from memory, reproduce common bugs, sweep settings, and validate.",
        "weight": 4,
        "notebook": "notebooks/labs/05_training_loop_lab.ipynb",
    },
]

LAB_STARTERS = {
    "02_tensor_lab.ipynb": [
        ("Prepare a batch", "t = torch.arange(12)\n\n# TODO: reshape to (3, 4) and (2, 2, 3), then flatten\n# Add an assertion after each operation."),
        ("Test broadcasting", "a = torch.zeros(3, 1)\nb = torch.zeros(4)\n\n# TODO: predict and assert the shape of a + b\n# TODO: add a length-3 bias to a (5, 3) batch"),
        ("Inspect non-contiguous storage", "t = torch.arange(6).reshape(2, 3)\nprint('t stride:', t.stride())\nprint('t.T stride:', t.T.stride())\n\n# TODO: reproduce the view error, then fix it two ways"),
        ("Remove an element loop", "import time\nx = torch.randn(100_000)\n\n# TODO: time a Python element loop and x.sum()\n# Assert that the answers match, then print the speedup."),
        ("Make dtype and device explicit", "device = (\n    'cuda' if torch.cuda.is_available() else\n    'mps' if torch.backends.mps.is_available() else\n    'cpu'\n)\nprint('device:', device)\n\n# TODO: move a small model and batch to this device and run them"),
    ],
    "03_autograd_lab.ipynb": [
        ("Predict and verify", "x = torch.tensor(2.0, requires_grad=True)\ny = x**3 + 2*x\n\n# Predict dy/dx first, then call backward and assert x.grad."),
        ("Inspect a graph", "x = torch.tensor(2.0, requires_grad=True)\nw = torch.tensor(3.0, requires_grad=True)\ny = (x * w + 1) ** 2\n\n# TODO: print grad_fn and next_functions, then verify both gradients"),
        ("Break and fix accumulation", "w = torch.tensor(1.0, requires_grad=True)\n\n# TODO: call backward three times without clearing w.grad\n# Then repeat while setting w.grad = None before each backward."),
        ("Compare no_grad and detach", "x = torch.tensor(3.0, requires_grad=True)\n\n# TODO: create one result inside torch.no_grad()\n# and one result with (x * 2).detach(); inspect requires_grad."),
        ("Write gradcheck", "def numerical_gradient(fn, x, eps=1e-4):\n    # TODO: central difference (f(x + eps) - f(x - eps)) / (2 * eps)\n    pass\n\n# Compare your result with autograd for f(x) = sin(x) * x**2."),
    ],
    "04_module_lab.ipynb": [
        ("Fix parameter registration", "class Broken(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.weight = torch.randn(3, 2)  # TODO: register this\n\nmodel = Broken()\nprint(list(model.named_parameters()))"),
        ("Save and restore state", "model = nn.Linear(2, 1)\noriginal = {k: v.clone() for k, v in model.state_dict().items()}\n\n# TODO: change the weights, reload original, and assert equality"),
        ("See train and eval modes", "dropout = nn.Dropout(p=0.8)\nx = torch.ones(10)\n\n# TODO: run in train mode twice and eval mode twice; compare"),
        ("Build a nested MLP", "class TinyMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # TODO: Linear(4, 8), ReLU, Linear(8, 2)\n\n    def forward(self, x):\n        # TODO\n        pass\n\n# Assert output shape for a (5, 4) batch."),
        ("Inspect optimizer references", "model = nn.Linear(2, 1)\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n\n# TODO: prove the optimizer holds references to the model parameters."),
    ],
    "05_training_loop_lab.ipynb": [
        ("Train a line from memory", "torch.manual_seed(0)\nx = torch.linspace(-1, 1, 100).reshape(-1, 1)\ny = 3 * x - 0.5\nmodel = nn.Linear(1, 1)\noptimizer = torch.optim.SGD(model.parameters(), lr=0.1)\nloss_fn = nn.MSELoss()\n\n# TODO: write the five-line training loop and reach loss < 1e-4"),
        ("Reproduce the zero_grad bug", "# TODO: train once with optimizer.zero_grad() and once without it.\n# Record both loss curves and compare their final values."),
        ("Sweep learning rates", "learning_rates = [0.001, 0.01, 0.1, 1.0]\nresults = {}\n\n# TODO: train a fresh model for each rate and store final loss."),
        ("Compare batch sizes", "from torch.utils.data import DataLoader, TensorDataset\n\n# TODO: train with batch sizes 1, 10, and 100.\n# Count optimizer steps and compare final loss."),
        ("Add validation", "# TODO: split the data 80/20. Train on 80%.\n# For validation, use model.eval() and torch.no_grad()."),
    ],
}


def front_matter(title: str, description: str, weight: int = 0, notebook: str | None = None,
                 completable: bool = True) -> str:
    values = [
        "+++",
        f"title = {json.dumps(title)}",
        f"description = {json.dumps(description)}",
        f"weight = {weight}",
        f"completable = {str(completable).lower()}",
    ]
    if notebook:
        values.append(f"notebook = {json.dumps(notebook)}")
    values += ["+++", ""]
    return "\n".join(values)


def strip_first_heading(text: str) -> str:
    return re.sub(r"\A\s*# .+?\n+", "", text, count=1)


def clean_markdown(text: str) -> str:
    text = re.sub(r"^> \*\*Workshop position:\*\*.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\*Next:\*\*.*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\*PyTorch Foundations.*?\*\*\s*$", "", text, flags=re.MULTILINE)
    replacements = {
        "01_what_is_pytorch.ipynb": "../01-what-is-pytorch/",
        "02_tensors_properly.ipynb": "../02-inside-a-tensor/",
        "03_autograd_without_magic.ipynb": "../03-autograd-without-magic/",
        "04_nn_module_and_parameters.ipynb": "../04-modules-and-parameters/",
        "05_the_training_loop.ipynb": "../05-training-loop/",
        "../projects/project1_linear_from_scratch.py": "#start-project",
        "../projects/project2_mlp_from_scratch.py": "#start-project",
        "../exercises/": "../../practice/",
        "../docs/CHEATSHEET.md": "../../reference/cheatsheet/",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def output_text(outputs: list[dict]) -> str:
    chunks: list[str] = []
    for output in outputs:
        if output.get("output_type") == "stream":
            value = output.get("text", "")
        elif output.get("output_type") in {"execute_result", "display_data"}:
            value = output.get("data", {}).get("text/plain", "")
        elif output.get("output_type") == "error":
            value = "\n".join(output.get("traceback", []))
        else:
            value = ""
        if isinstance(value, list):
            value = "".join(value)
        value = re.sub(r"\x1b\[[0-9;]*m", "", value).strip()
        if value:
            chunks.append(value)
    return "\n".join(chunks)


def notebook_body(path: Path) -> str:
    notebook = json.loads(path.read_text())
    pieces: list[str] = []
    first_markdown = True
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", [])).strip()
        if not source:
            continue
        if cell["cell_type"] == "markdown":
            if first_markdown:
                source = strip_first_heading(source)
                first_markdown = False
            source = clean_markdown(source)
            if source:
                pieces.append(source)
        elif cell["cell_type"] == "code":
            pieces.append(f"```python\n{source}\n```")
            rendered = output_text(cell.get("outputs", []))
            if rendered:
                rendered = rendered.replace("{{", "{ {").replace("}}", "} }")
                pieces.append("{{< output >}}\n" + rendered + "\n{{< /output >}}")
    return "\n\n".join(pieces).strip() + "\n"


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = "\n".join(line.rstrip() for line in value.splitlines())
    path.write_text(value.rstrip() + "\n")


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def save_notebook(path: Path, cells: list[dict]) -> None:
    data = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def create_practice_labs() -> None:
    intro = code_cell("import torch\nimport torch.nn as nn\nprint('PyTorch:', torch.__version__)")
    for item in PRACTICE:
        filename = Path(item["notebook"]).name
        cells = [
            markdown_cell(
                f"# {item['title']}\n\n"
                "Run the setup cell, complete one challenge at a time, and keep every assertion green. "
                "On a phone, use the **⋮** menu to save a copy to Drive first."
            ),
            intro,
        ]
        for index, (heading, starter) in enumerate(LAB_STARTERS[filename], start=1):
            cells.append(markdown_cell(f"## {index}. {heading}\n\nPredict the result before you run the cell."))
            cells.append(code_cell(starter))
        cells.append(markdown_cell("## Finish line\n\nRestart the runtime and run all cells. You are done when every cell finishes without an error."))
        save_notebook(ROOT / item["notebook"], cells)


def top_level_nodes(source: str) -> list[tuple[str, str]]:
    lines = source.splitlines()
    tree = ast.parse(source)
    sections: list[tuple[str, str]] = []
    imports: list[str] = []
    for node in tree.body:
        block = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(block)
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            sections.append((node.name, block))
    return [("Imports", "\n".join(imports))] + sections


def create_project_labs() -> None:
    for item in PROJECTS:
        source = (ROOT / item["starter"]).read_text()
        cells = [
            markdown_cell(
                f"# {item['title']}\n\n"
                "Complete each TODO, run its checkpoint, and do not open the answer until you have a failing attempt. "
                "Use **Runtime → Run all** for the final check."
            )
        ]
        for name, block in top_level_nodes(source):
            if name == "main":
                cells.append(markdown_cell("## Final training check"))
            elif name == "Imports":
                cells.append(markdown_cell("## Setup"))
            else:
                cells.append(markdown_cell(f"## Implement `{name}`"))
            cells.append(code_cell(block))
        cells.append(markdown_cell("## Run every checkpoint"))
        cells.append(code_cell("main()"))
        save_notebook(ROOT / item["notebook"], cells)


def build_sections() -> None:
    write(
        CONTENT / "_index.md",
        front_matter(
            "PyTorch Foundations",
            "A practical, mobile-first PyTorch workshop with runnable lessons and from-scratch projects.",
            completable=False,
        ),
    )
    write(
        CONTENT / "modules/_index.md",
        front_matter(
            "Modules",
            "Five short lessons. Read a little, run immediately, then prove the idea with a checkpoint.",
            completable=False,
        ) + "Follow these in order. Each one reuses the implementation pattern from the previous lesson.",
    )
    write(
        CONTENT / "practice/_index.md",
        front_matter(
            "Practice labs",
            "Open a starter in Colab, implement five focused challenges, then reveal the runnable answer.",
            completable=False,
        ) + "Attempt each lab after its matching module. The answer is collapsed at the bottom of each page.",
    )
    write(
        CONTENT / "projects/_index.md",
        front_matter(
            "Build projects",
            "Two end-to-end builds that turn tensors, gradients, modules, and training loops into working systems.",
            completable=False,
        ) + "Start from the Colab scaffold. Use the walkthrough only after your first implementation attempt.",
    )
    write(
        CONTENT / "reference/_index.md",
        front_matter("Reference", "Keep the core PyTorch patterns within thumb's reach.", completable=False),
    )


def build_getting_started() -> None:
    body = r"""
You need a browser and a Google account. You do **not** need Python, a laptop, or a GPU.

## Your 3-minute setup {#phone}

1. Tap **Open in Colab** on any lesson or lab.
2. In Colab, open the **⋮** menu and choose **Save a copy in Drive**.
3. Tap the play icon beside the first code cell. If Google asks, choose **Run anyway**.
4. Change one value and run the cell again. That saved copy is your working notebook.

> On a small screen, rotate to landscape for long code cells. Pinch to zoom out once if a table feels cramped.

## The workshop route

| Time | Do this | Proof you are done |
|---|---|---|
| 0:00–0:20 | Run the PyTorch overview | You can explain tensor → loss → gradient → update |
| 0:20–1:50 | Complete Modules 2–5 and their labs | Your assertions pass and your model loss falls |
| 1:50–2:30 | Build your own linear layer | It matches `nn.Linear` and learns `y = 2.5x + 1` |
| 2:30–3:15 | Build the XOR network | Your MLP classifies all four XOR points |
| 3:15–3:30 | Break, fix, recap | You can diagnose one shape or gradient bug |

## How to use every page

Use the same four moves throughout the workshop:

1. **Predict** the shape, value, or gradient.
2. **Run** the code.
3. **Change** one thing and run it again.
4. **Prove** the result with an assertion or checkpoint.

Start with [What PyTorch actually does](../modules/01-what-is-pytorch/). Keep the [cheatsheet](../reference/cheatsheet/) open in another tab.
"""
    write(
        CONTENT / "getting-started.md",
        front_matter(
            "Start here",
            "Set up Google Colab on your phone, understand the workshop route, and run your first cell.",
            weight=0,
            completable=False,
        ) + body,
    )


def build_modules_and_projects() -> None:
    for item in MODULES:
        body = notebook_body(ROOT / item["source"])
        text = front_matter(
            item["title"], item["description"], item["weight"], notebook=item["source"]
        ) + body
        write(ROOT / item["target"], text)

    for item in PROJECTS:
        body = notebook_body(ROOT / item["source"])
        addition = f"""

## Start project

Open the Colab scaffold above and complete the TODOs. If you prefer a local file, use the [Python starter]({{{{< param githubRepo >}}}}/blob/main/{item['starter']}).

{{{{< details summary="Reveal the complete reference implementation" >}}}}

```python
{(ROOT / item['answer']).read_text().rstrip()}
```

{{{{< /details >}}}}
"""
        text = front_matter(
            item["title"], item["description"], item["weight"], notebook=item["notebook"]
        ) + body + addition
        write(ROOT / item["target"], text)


def build_practice() -> None:
    for item in PRACTICE:
        body = strip_first_heading((ROOT / item["source"]).read_text())
        body = re.sub(r"Runnable solutions:.*?\.\n", "", body)
        answer = (ROOT / item["answer"]).read_text().rstrip()
        addition = f"""

{{{{< details summary="Reveal the runnable solution" >}}}}

```python
{answer}
```

{{{{< /details >}}}}
"""
        write(
            ROOT / item["target"],
            front_matter(item["title"], item["description"], item["weight"], notebook=item["notebook"])
            + body + addition,
        )


def build_reference() -> None:
    cheatsheet = strip_first_heading((ROOT / "docs/CHEATSHEET.md").read_text())
    write(
        CONTENT / "reference/cheatsheet.md",
        front_matter(
            "PyTorch implementation cheatsheet",
            "Shapes, devices, gradients, modules, loops, validation, and debugging in one compact page.",
            weight=1,
            completable=False,
        ) + cheatsheet,
    )


def main() -> None:
    if CONTENT.exists():
        shutil.rmtree(CONTENT)
    build_sections()
    build_getting_started()
    create_practice_labs()
    create_project_labs()
    build_modules_and_projects()
    build_practice()
    build_reference()
    print(f"Built Hugo content in {CONTENT}")
    print(f"Built Colab labs in {LABS}")


if __name__ == "__main__":
    main()
