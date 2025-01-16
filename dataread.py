import pandas as pd
import serial
import matplotlib.pyplot as plt
import wave
import numpy as np

# UART Configuration
serial_port = 'COM3'  # Replace with your UART port (e.g., '/dev/ttyUSB0' for Linux)
baud_rate = 921600  # Set the baud rate to your UART module

# Initialize serial communication
ser = serial.Serial(serial_port, baud_rate, timeout=1)
print(f"Connected to {serial_port} at {baud_rate} baud.")

data = []

# Prepare to write the received data into a .wav file (mono audio)
output_file = wave.open('received_audio.wav', 'wb')
output_file.setnchannels(1)  # Mono channel
output_file.setsampwidth(2)  # 16-bit samples (2 bytes per sample)
output_file.setframerate(44100)  # 44.1kHz sample rate (or adjust as needed)

# # Function to update the plot
# def update_plot():
#     plt.clf()
#     plt.plot(data)
#     plt.title("UART Data")
#     plt.xlabel("Sample")
#     plt.ylabel("Value")
#     plt.draw()
#     plt.pause(0.001)  # Pause to allow the plot to update

# # Initialize the plot
# plt.ion()
# fig = plt.figure()

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            # Split the line into individual values
            values = line.split()
            for value in values:
                try:
                    data.append(float(value))  # Convert to float for plotting
                    print(value)  # Print the data to the console (optional)
                except ValueError as e:
                    print(f"Could not convert value to float: {value}")

            # # Update the plot with the new data
            # update_plot()

            # Check if we have collected 100,000 data points
            data_points = 10000
            if len(data) >= data_points:
                print(f"Collected {data_points} data points. Stopping...")
                break

except KeyboardInterrupt:
    print("Keyboard interrupt detected. Stopping...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ser.close()
    print("Serial connection closed.")

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=['Data'])

    # Save the DataFrame to an Excel file
    df.to_excel('data.xlsx', index=False)
    print("Data saved to data.xlsx")

     # Save the audio data to a WAV file
    audio_data_np = np.array(data, dtype=np.uint16)
    output_file.writeframes(audio_data_np.tobytes())
    output_file.close()
    print("Audio saved to received_audio.wav")