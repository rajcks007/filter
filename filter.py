import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd


# Load data from the Excel file
file_path = 'Low_pass.xlsx'  # Replace with the path to your file
data = pd.read_excel(file_path, header=None)  # No header in the file
# sheet_name = 'Sheet1'  # Replace with the sheet name if needed

# Assign a temporary column name
data.columns = ['Signal']  # Rename the single column to 'Signal'

print(data.head())

# Extract the signal as a numpy array (if needed)
signal = data['Signal'].to_numpy()

# Filter specifications
order = 4               # Filter order
cutoff_low = 10         # Lower cutoff frequency in Hz
cutoff_high = 40        # Upper cutoff frequency in Hz
fs = 900               # Sampling frequency in Hz

# Design Butterworth filter
b, a = butter(order, cutoff_low / (fs / 2), btype='low')  # 'low', 'high', 'bandpass', or 'bandstop'

# Design Butterworth filter
# b, a = butter(order, cutoff_high / (fs / 2), btype='high')  # 'low', 'high', 'bandpass', or 'bandstop'

# Design Butterworth band-pass filter
# b, a = butter(order, [cutoff_low / (fs / 2), cutoff_high / (fs / 2)], btype='bandpass')  # Band-pass filter

# Design Butterworth band-stop filter
# b, a = butter(order, [cutoff_low / (fs / 2), cutoff_high / (fs / 2)], btype='bandstop')  # Band-stop filter

# Apply the filter to the data using filtfilt
# b, a: Numerator (b) and denominator (a) polynomials of the IIR filter.
filtered_signal = filtfilt(b, a, signal)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(signal, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal', linestyle='--')
plt.title('Original vs Filtered Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

print("Numerator coefficients (b):", b)
print("Denominator coefficients (a):", a)