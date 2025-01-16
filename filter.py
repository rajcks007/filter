import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter, medfilt, freqz
import matplotlib.pyplot as plt
import pandas as pd


# Load data from the Excel file
file_path = 'data.xlsx'  # Replace with the path to your file
data = pd.read_excel(file_path, header=0)  # No header in the file
# sheet_name = 'Sheet1'  # Replace with the sheet name if needed

# Assign a temporary column name
data.columns = ['Signal']  # Rename the single column to 'Signal'

# Take only the first 100 data points
data = data.head(1000)

print(data.head())

# Extract the signal as a numpy array (if needed)
signal = data['Signal'].to_numpy()

# ----------------------------
# Median Filter
# ----------------------------
window_size_median = 5  # Odd number for median filter window size
median_filtered_signal = medfilt(signal, kernel_size=window_size_median)

# ----------------------------
# Savitzky-Golay Filter
# ----------------------------
window_size_savgol = 7  # Odd number, size of the window
poly_order_savgol = 2   # Polynomial order (must be < window_size)
try:
    savgol_filtered_signal = savgol_filter(signal, window_length=window_size_savgol, polyorder=poly_order_savgol)
except ValueError as e:
    print(f"Error with Savitzky-Golay filter: {e}")
    exit()

# ----------------------------
# Butterworth Filter (High-pass example)
# ----------------------------

# Filter specifications
order = 1               # Filter order
cutoff_low = 1000         # Lower cutoff frequency in Hz
cutoff_high = 3550        # Upper cutoff frequency in Hz
fs = 7200              # Sampling frequency in Hz

# Design Butterworth filter
b, a = butter(order, cutoff_low / (fs / 2), btype='low')  # low-pass filter

# Design Butterworth filter
# b, a = butter(order, cutoff_high / (fs / 2), btype='high')  # high-pass filter

# Design Butterworth band-pass filter
# b, a = butter(order, [cutoff_low / (fs / 2), cutoff_high / (fs / 2)], btype='bandpass')  # Band-pass filter

# Design Butterworth band-stop filter
# b, a = butter(order, [cutoff_low / (fs / 2), cutoff_high / (fs / 2)], btype='bandstop')  # Band-stop filter

# Apply the filter to the data using filtfilt
# b, a: Numerator (b) and denominator (a) polynomials of the IIR filter.
filtered_signal = filtfilt(b, a, signal)

# ----------------------------
# Plot Results
# ----------------------------

print("Numerator coefficients (b):", b)

print("Denominator coefficients (a):", a)

w, h = freqz(b, a, worN=8000)

# Plot the original and filtered signals
plt.figure(1, figsize=(10, 6))
plt.plot(signal, label='Original Signal')
plt.plot(filtered_signal, label='Filtered Signal')
plt.title('Original vs Filtered Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Plot Magnitude Response
# plt.figure(2, figsize=(10, 6))
# plt.plot((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)), label='Magnitude Response')
# plt.axvline(cutoff_low, color='red', linestyle='--', label=f'Cutoff Frequency: cutoff_low Hz')
# plt.axvline(cutoff_high, color='red', linestyle='--', label=f'Cutoff Frequency: cutoff_high Hz')
# plt.title('Butterworth Filter Frequency Response')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude (dB)')
# plt.grid()
# plt.legend()

# Show all plots at once
plt.show()