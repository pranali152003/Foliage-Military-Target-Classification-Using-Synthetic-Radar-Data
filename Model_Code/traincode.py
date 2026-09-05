import pandas as pd
import matplotlib.pyplot as plt

from build_model import model
from load_dataset import train_dataset, validation_dataset

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger
)

# -----------------------------
# Callbacks
# -----------------------------

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_radar_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

csv_logger = CSVLogger(
    "training_history.csv",
    append=False
)

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr,
        csv_logger
    ]
)

# -----------------------------
# Save Final Model
# -----------------------------

model.save("radar_cnn_model.keras")

print("\nModel saved successfully!")

# -----------------------------
# Save History
# -----------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    "history.csv",
    index=False
)

print("History saved successfully!")

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Training vs Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig(
    "accuracy_graph.png",
    dpi=300
)

plt.show()

# -----------------------------
# Loss Graph
# -----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Training vs Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig(
    "loss_graph.png",
    dpi=300
)

plt.show()

print("\nGraphs Saved Successfully!")