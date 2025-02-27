import tensorflow as tf
import numpy

print(tf.__version__)

# Example model 
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2048, 1)),  # Assuming 2048-point waveforms
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])


