import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load audio file
audio_file = "received_audio.wav"  # Replace with your file path
y, sr = librosa.load(audio_file, sr=None)  # Load with original sample rate

# Generate Short-Time Fourier Transform (STFT)
D = librosa.stft(y)  # STFT of the audio signal
spectrogram = np.abs(D)  # Magnitude of the STFT

# Convert the magnitude spectrogram to decibels
spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

# # Display the spectrogram
# plt.figure(figsize=(10, 6))
# librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (Log Frequency)')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.show()

# Compute Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to dB
mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

# Normalize the spectrogram
spectrogram_normalized = (mel_spectrogram_db - np.min(mel_spectrogram_db)) / \
                         (np.max(mel_spectrogram_db) - np.min(mel_spectrogram_db))

plt.figure(figsize=(10, 6))
librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Mel Frequency')
plt.show()


# Reshape for TensorFlow input
# Add a channel dimension if using CNNs
spectrogram_tensor = np.expand_dims(spectrogram_normalized, axis=-1)  # Shape: (128, TimeSteps, 1)

# Save the spectrogram with a label
data = {'spectrogram': spectrogram_normalized, 'label': 'noise'}
with open('file_2.pkl', 'wb') as f:
    pickle.dump(data, f)
