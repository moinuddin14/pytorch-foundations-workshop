+++
title = "Start here"
description = "Set up Google Colab on your phone, understand the workshop route, and run your first cell."
weight = 0
completable = false
+++

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
