from flask import Flask, request, jsonify
import torch
import torch.nn as nn

app = Flask(__name__)

# Define same model structure
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

model.load_state_dict(torch.load("study_model.pth"))
model.eval()

@app.route("/")
def home():
    return "AI Study Planner Backend Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    hours_available = float(data["hours_available"])
    subjects_count = float(data["subjects_count"])

    input_data = torch.tensor([[hours_available, subjects_count]])

    with torch.no_grad():
        prediction = model(input_data)

    return jsonify({
        "predicted_hours_per_subject": prediction.item()
    })

# ✅ IMPORTANT: ADD TEST ROUTE HERE (ABOVE app.run)

@app.route("/test")
def test():
    input_data = torch.tensor([[5.0, 2.0]])

    with torch.no_grad():
        prediction = model(input_data)

    return jsonify({
        "predicted_hours_per_subject": prediction.item()
    })

# THIS MUST BE LAST
if __name__ == "__main__":
    app.run(debug=True)

