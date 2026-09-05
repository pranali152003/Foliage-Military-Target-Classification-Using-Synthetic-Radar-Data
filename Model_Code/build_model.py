import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------
# Load MobileNetV2
# ------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# ------------------------------------------------
# Build Model
# ------------------------------------------------

model = Sequential([

    Input(shape=(224,224,3)),

    # Convert pixel values 0-255 → 0-1
    tf.keras.layers.Rescaling(1./255),

    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.4),

    Dense(
        4,
        activation='softmax'
    )

])

# ------------------------------------------------
# Compile
# ------------------------------------------------

model.compile(

    optimizer=Adam(
        learning_rate=0.0001
    ),

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

model.summary()

print("MobileNetV2 Model Ready!")