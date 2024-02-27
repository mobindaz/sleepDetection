import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model, optimizers

# Step 1: Organize Image Dataset
dataset_path = "path/to/dataset"
class_names = os.listdir(dataset_path)

# Step 2: Data Loading and Preprocessing
images = []
labels = []
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Resize image if needed
        images.append(image)
        labels.append(class_name)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Preprocess images (e.g., normalize pixel values)
images = images / 255.0

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Step 3: Model Definition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Step 4: Model Training
model.compile(optimizer=optimizers.Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Step 5: Model Evaluation (optional)
loss, accuracy = model.evaluate(val_images, val_labels)
print("Validation Accuracy:", accuracy)

# Step 6: Save the trained model to disk
model_save_path = "model.h5"
model.save(model_save_path)

print("Model saved successfully at:", model_save_path)
