import torch
import torch.nn as nn
import torch.optim as optim

# Dummy dataset: [hours_available, subjects_count]
X = torch.tensor([
    [2, 3],
    [5, 2],
    [1, 4],
    [6, 1],
    [3, 3],
    [4, 2]
], dtype=torch.float32)

# Target: predicted study hours per subject
y = torch.tensor([
    [0.6],
    [2.5],
    [0.25],
    [6.0],
    [1.0],
    [2.0]
], dtype=torch.float32)

# Define simple neural network
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(500):
    predictions = model(X)
    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training Complete")

# Save the model
torch.save(model.state_dict(), "study_model.pth")
print("Model saved as study_model.pth")
