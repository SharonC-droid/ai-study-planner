import torch
import torch.nn as nn

# Define same model structure
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# Load trained weights
model.load_state_dict(torch.load("study_model.pth"))
model.eval()

# Example input:
# [hours_available, subjects_count]
input_data = torch.tensor([[5.0, 2.0]])

with torch.no_grad():
    prediction = model(input_data)

print("Predicted study hours per subject:", prediction.item())
