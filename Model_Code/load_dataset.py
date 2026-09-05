import tensorflow as tf

# Dataset Paths
train_path = r"C:\Users\admin\Desktop\DIAT\Radar_Dataset\train"
validation_path = r"C:\Users\admin\Desktop\DIAT\Radar_Dataset\validation"
test_path = r"C:\Users\admin\Desktop\DIAT\Radar_Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Load Validation Dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load Test Dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nClass Names:")
print(train_dataset.class_names)

print("\nTraining Batches:", len(train_dataset))
print("Validation Batches:", len(validation_dataset))
print("Testing Batches:", len(test_dataset))

# Dataset loading is only for preparing the data.
#We used TensorFlow's image_dataset_from_directory() function to load the radar image dataset
#  from the train, validation, and test folders into memory so it can be used by the CNN model.
#  During this process, TensorFlow automatically reads all the images from their respective 
# class folders, resizes them to 224 × 224 pixels, assigns labels based on the folder names (such as human, machine_gun, tank, and vehicle),
#  groups the images into batches (e.g., 32 images per batch), and prepares them in a format (tensors) that the CNN can understand and process efficiently. 
# This step is essential because the CNN cannot directly read image files or folders from the disk; it requires the data to be converted into numerical tensors with corresponding labels before training can begin.