import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Description: This file contains the implementation of the AI model. The AI model is a denoising autoencoder that is used to remove noise from the received audio data. The model is built using the TensorFlow library and Keras API. The model architecture consists of convolutional layers for feature extraction and upsampling layers for reconstruction. The model is trained using the mean squared error loss function and the Adam optimizer. The model is then used to denoise the received audio data by passing it through the autoencoder. The denoised audio data is then saved to a new file for further analysis.
import tensorflow
from tensorflow.python.keras import models , layers
import pickle
import numpy as np

# Load multiple files (replace with your file iteration logic)
for file in ['file_1.pkl', 'file_2.pkl']:  # Replace with actual file paths
    with open(file, 'rb') as f:
        data = pickle.load(f)
    spectrogram = data['spectrogram']
    label = data['label']

# Map string labels to integers
label_mapping = {'clean': 0, 'noise': 1}
if label in label_mapping:
    label = label_mapping[label]
else:
    raise ValueError(f"Unknown label: {label}")

# Prepare spectrogram and label
spectrogram_tensor = tensorflow.convert_to_tensor(spectrogram, dtype=tensorflow.float32)  # Convert to Tensor
spectrogram_tensor = tensorflow.expand_dims(spectrogram_tensor, axis=-1)  # Add channel dimension

try:
    label_tensor = tensorflow.convert_to_tensor(label, dtype=tensorflow.int32)
except Exception as e:
    print(f"Error converting label to tensor: {e}")

# Combine into a single dataset
dataset = tensorflow.data.Dataset.from_tensors((spectrogram_tensor, label_tensor))

'''
Prepare the Dataset for Training
TensorFlow datasets require preprocessing for efficient training:

1. Shuffle the data to avoid ordering bias.
2. Batch the data for training.
3. Prefetch to improve performance.
'''
# Preprocess the dataset
BATCH_SIZE = 32

dataset = dataset.shuffle(buffer_size=100)  # Shuffle the data
dataset = dataset.batch(BATCH_SIZE)  # Create batches
dataset = dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)  # Prefetch for performance


# Define a simple CNN model
input_shape = (128, None, 1)  # Mel bins, time steps (dynamic), channel

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Print model summary
model.summary()

# #  Train the Model
# model.fit(dataset, epochs=50, batch_size=32)

# Convert to TensorFlow Lite
converter = tensorflow.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model
with open('denoising_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Quantize the Model for Embedded Use
converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Save the quantized model
with open('denoising_model_quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)