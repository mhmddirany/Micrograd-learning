from micrograd.engine import Value
from micrograd.nn import MLP

# --- Part 1: automatic gradients on a tiny expression ---
a = Value(2.0)
b = Value(-3.0)
c = a * b + a**2          # c = -6 + 4 = -2
c.backward()              # compute dc/da and dc/db
print(f"c = {c.data}")
print(f"dc/da = {a.grad}  (expected b + 2a = 1)")
print(f"dc/db = {b.grad}  (expected a = 2)")

# --- Part 2: train a tiny neural network ---
xs = [[2, 3, -1], [3, -1, 0.5], [0.5, 1, 1], [1, 1, -1]]
ys = [1, -1, -1, 1]
model = MLP(3, [4, 4, 1])

for step in range(50):
    preds = [model(x) for x in xs]
    loss = sum((p - y)**2 for p, y in zip(preds, ys))
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= 0.05 * p.grad
    if step % 10 == 0 or step == 49:
        print(f"step {step:2d}  loss {loss.data:.4f}")

print("predictions:", [round(model(x).data, 2) for x in xs])
print("targets:    ", ys)