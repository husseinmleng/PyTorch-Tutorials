import torch
from torch.autograd import Variable
# Create a tensor and set requires_grad=True to track computation with it
x = torch.randn(3,requires_grad=True)
print(x)
# Do a tensor operation:
y = x + 2
print(y)
# y was created as a result of an operation, so it has a grad_fn.
z = y * y * 2
z = z.mean()
print(z)

z.backward()
print(x.grad)
# create a trainging example
weights = torch.randn(4,requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()

# optimizer training
optimizer = torch.optim.SGD([weights],lr=0.01)
optimizer.step()
optimizer.zero_grad()
