import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# =====================================================
# 1. Load Fine-Tuned Model
# =====================================================

model = load_model(
    r"C:\Users\admin\Desktop\DIAT\best_radar_model.keras"
)

print("=" * 60)
print("       RADAR OBJECT CLASSIFICATION SYSTEM")
print("=" * 60)
print("Model Loaded Successfully!\n")


# =====================================================
# 2. Class Names
# =====================================================

class_names = [
    "Human",
    "Machine Gun",
    "Tank",
    "Vehicle"
]


# =====================================================
# 3. Image Path
# =====================================================

img_path = r"C:\Users\admin\Desktop\aa.jpg"


# =====================================================
# 4. Load Image
# =====================================================

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(img).astype("float32")

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)


# =====================================================
# IMPORTANT
# =====================================================
# Your model already contains:
#
# Rescaling(1./255)
#
# Therefore DON'T do:
#
# MobileNetV2 preprocess_input()
#
# Keep image values between 0-255 here.


# =====================================================
# 5. Prediction
# =====================================================

prediction = model.predict(
    img_array,
    verbose=0
)

prediction = prediction[0]


# =====================================================
# 6. Check Prediction Vector
# =====================================================

print("Prediction Vector:")
print(prediction)

print("\nProbability Sum:")
print(np.sum(prediction))


# =====================================================
# 7. Get Prediction
# =====================================================

predicted_index = np.argmax(prediction)

predicted_class = class_names[predicted_index]

confidence = prediction[predicted_index] * 100


# =====================================================
# 8. Terminal Output
# =====================================================

print("\n" + "=" * 60)
print("Prediction Result")
print("=" * 60)

print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence:.2f}%")


print("\nProbability of Each Class")
print("-" * 60)

for cls, prob in zip(class_names, prediction):

    print(
        f"{cls:<15} : {prob * 100:.2f}%"
    )


# =====================================================
# 9. Visualization
# =====================================================

fig = plt.figure(figsize=(13, 5))


# -----------------------------------------------------
# Image
# -----------------------------------------------------

plt.subplot(1, 2, 1)

plt.imshow(img)

plt.axis("off")

plt.title(
    f"Prediction\n"
    f"{predicted_class}\n"
    f"Confidence : {confidence:.2f}%",
    fontsize=14,
    fontweight="bold"
)


# -----------------------------------------------------
# Probability Graph
# -----------------------------------------------------

plt.subplot(1, 2, 2)

bars = plt.bar(
    class_names,
    prediction * 100
)

# Highlight predicted class

bars[predicted_index].set_color("green")


plt.ylim(0, 100)

plt.ylabel("Probability (%)")

plt.title("Prediction Probability")

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)


# Display percentages

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.1f}%",
        ha="center",
        fontweight="bold"
    )


plt.tight_layout()

plt.show()