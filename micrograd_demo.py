# micrograd_demo.py
# My first neural network, built with Andrej Karpathy's micrograd.
# Part 1 shows backpropagation on a tiny math expression.
# Part 2 trains a small neural network to match 4 target values.

import random
from micrograd.engine import Value   # a number that tracks its own gradient
from micrograd.nn import MLP         # a multi-layer neural network

# Fix the randomness so every run starts from the same weights.
# This lets me compare experiments fairly. Change 42 to try another start.
random.seed(42)

# ---------------------------------------------------------------
# Part 1: automatic gradients (backpropagation)
# ---------------------------------------------------------------
a = Value(2.0)
b = Value(-3.0)
c = a * b + a**2      # c = (2 * -3) + 2^2 = -6 + 4 = -2

# backward() walks the computation in reverse using the chain rule,
# filling in .grad = how much c changes if we nudge that value.
c.backward()

print(f"c = {c.data}")
print(f"dc/da = {a.grad}  (expected b + 2a = 1)")
print(f"dc/db = {b.grad}  (expected a = 2)")

# ---------------------------------------------------------------
# Part 2: train a tiny neural network
# ---------------------------------------------------------------
# Training data: 4 examples, each with 3 input numbers.
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1, -1, -1, 1]   # the answer we want for each example

# Network shape: 3 inputs -> 4 neurons -> 4 neurons -> 1 output.
# Hidden neurons use ReLU (negative values become 0).
model = MLP(3, [4, 4, 1])

# Learning rate: how big a step each update takes.
#   0.5  = too big, the network can "die" (all predictions 0)
#   0.05 = learns, but the loss bounces around
#   0.01 = slow but smooth
learning_rate = 0.01
steps = 200

for step in range(steps):
    # 1. Forward pass: run every example through the network.
    preds = [model(x) for x in xs]

    # 2. Loss: sum of squared errors. Lower is better, 0 is perfect.
    loss = sum((p - y)**2 for p, y in zip(preds, ys))

    # 3. Backward pass: reset old gradients, then compute new ones.
    model.zero_grad()
    loss.backward()

    # 4. Update: nudge every weight a little in the direction
    #    that makes the loss smaller (gradient descent).
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # Print progress every 20 steps and at the end.
    if step % 20 == 0 or step == steps - 1:
        print(f"step {step:3d}  loss {loss.data:.4f}")

# Compare what the network learned against the targets.
print("predictions:", [round(model(x).data, 2) for x in xs])
print("targets:    ", ys)