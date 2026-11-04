from pathlib import Path

import joblib
import torch
from torch import nn

from src.dataset import prepare_data
from src.classifier import GestureClassifier


# Configuration
EPOCHS = 100
LEARNING_RATE = 0.001  # Step size h for Euler's method
PATIENCE = 12
MODEL_DIR = Path("models")
SCALER_PATH = MODEL_DIR / "scaler.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load datasets
train_loader, val_loader, test_loader, encoder, scaler = prepare_data()
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved at: {SCALER_PATH}")

num_classes = len(encoder.classes_)

print("Classes:", list(encoder.classes_))
print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))


# Create model
model = GestureClassifier(num_classes=num_classes).to(device)


# Loss function
criterion = nn.CrossEntropyLoss()


# Training history
history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_loss = float("inf")
patience_counter = 0

checkpoint_path = MODEL_DIR / "gesture_model.pth"


# Euler parameter update
def euler_step(model, h):
    """
    Update model parameters using the explicit Euler method.

    Continuous gradient-flow equation:
        d(theta)/dt = -gradient(L)

    Euler discretization:
        theta_next = theta - h * gradient(L)
    """
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter -= h * parameter.grad


# Training loop
for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss_total = 0.0

    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        # Reset previous gradients
        model.zero_grad(set_to_none=True)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Explicit Euler update
        euler_step(model, LEARNING_RATE)

        train_loss_total += loss.item() * features.size(0)

    train_loss = train_loss_total / len(train_loader.dataset)


    # Validation
    model.eval()

    val_loss_total = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * features.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss = val_loss_total / len(val_loader.dataset)
    val_accuracy = correct / total


    # Record history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch + 1:03d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2%}"
    )


    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": encoder.classes_.tolist(),
                "input_size": 63,
                "num_classes": num_classes,
                "method": "explicit_euler",
                "step_size": LEARNING_RATE,
            },
            checkpoint_path
        )

        print("  Best Euler model saved.")

    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= PATIENCE:
        print("Early stopping: validation loss did not improve.")
        break


print(f"\nTraining completed.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved at: {checkpoint_path}")