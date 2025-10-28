import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Set style for better visualizations
plt.style.use('default')
sns.set_context("notebook", font_scale=1.2)

# Find the most recent data file
list_of_files = glob.glob('sensor_data_*.csv')
if not list_of_files:
    print("No data files found. Run data_logger.py first.")
    exit()

latest_file = max(list_of_files, key=os.path.getctime)
print(f"Processing the most recent file: {latest_file}")

# Load data
df = pd.read_csv(latest_file)

# Convert to numeric and drop NaN values
numeric_columns = ["Temperature", "Humidity", "Vibration", 
                   "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)

# Add timestamps as index if available
if "Timestamp" in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

# Feature Engineering
df["AccelMagnitude"] = np.sqrt(df["AccelX"]**2 + df["AccelY"]**2 + df["AccelZ"]**2)
df["GyroMagnitude"] = np.sqrt(df["GyroX"]**2 + df["GyroY"]**2 + df["GyroZ"]**2)
df["Temp_Change"] = df["Temperature"].diff().fillna(0)
df["Accel_Change"] = df["AccelMagnitude"].diff().fillna(0)

print("\nData Summary:")
print(df.describe())

# Check for abnormal values
print("\nChecking for outliers...")
for col in ["Temperature", "AccelMagnitude", "GyroMagnitude"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    if not outliers.empty:
        print(f"- Found {len(outliers)} outliers in {col}")

# Create failure conditions label
# Use both vibration sensor and accelerometer/gyro data
df["Failure"] = ((df["Temperature"] > df["Temperature"].quantile(0.85)) | 
                ((df["Vibration"] == 1) & 
                 (df["AccelMagnitude"] > df["AccelMagnitude"].quantile(0.7)) |
                 (df["GyroMagnitude"] > df["GyroMagnitude"].quantile(0.7)))).astype(int)

print(f"\nDetected {df['Failure'].sum()} potential failure events in dataset")

# Plot sensor readings
plt.figure(figsize=(10,8))

# Temperature and humidity
plt.subplot(3,1,1)
plt.plot(df.index, df["Temperature"], 'r-', label="Temperature (°C)")
plt.plot(df.index, df["Humidity"], 'b-', label="Humidity (%)")
plt.legend()
plt.title("Temperature and Humidity Over Time")
plt.grid(True)

# Vibration and acceleration
plt.subplot(3,1,2)
plt.plot(df.index, df["AccelMagnitude"], 'g-', label="Accel Magnitude")
plt.bar(df.index, df["Vibration"]*df["AccelMagnitude"].max()*0.5, color='purple', alpha=0.3, label="Vibration")
plt.legend()
plt.title("Vibration and Acceleration")
plt.grid(True)

# Gyroscope and failure detection
plt.subplot(3,1,3)
plt.plot(df.index, df["GyroMagnitude"], 'c-', label="Gyro Magnitude")
if df["Failure"].sum() > 0:
    plt.plot(df.index, df["Failure"]*df["GyroMagnitude"].max()*0.8, 'r*', label="Failure Flag")
plt.legend()
plt.title("Gyroscope Readings and Detected Failures")
plt.grid(True)

plt.tight_layout()
plt.savefig("sensor_readings.png", dpi=300)
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(10,8))
correlation = df[["Temperature", "Humidity", "Vibration", 
                 "AccelMagnitude", "GyroMagnitude", "Failure"]].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Between Sensor Readings")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# Save processed data
output_file = "processed_sensor_data.csv"
df.to_csv(output_file)
print(f"\nProcessed data saved to '{output_file}'")
print("Visualizations saved as PNG files")