import torch

# Create a simple 5D tensor
B = 1  # Batch size
C = 1  # Number of channels
W = 2  # Width and height of the 3D volume
x = torch.arange(B * C * W * W * W).reshape(B, C, W, W, W)

# Print the original tensor
print("Original Tensor:")
print(x)

# Perform the view and shift operations
x = x.view(B, C, W, W, W)
shift_amount = int(W / 2)
x = torch.roll(x, shifts=(shift_amount, shift_amount, shift_amount), dims=(2, 3, 4))

# Print the modified tensor
print("\nModified Tensor:")
print(x)





