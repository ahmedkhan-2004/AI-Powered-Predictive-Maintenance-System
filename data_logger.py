import serial
import csv
import time
import os
from datetime import datetime

def find_arduino_port():
    """Try to automatically find the Arduino port."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if 'Arduino' in p.description or 'CH340' in p.description:
                return p.device
        return None
    except:
        return None

# Create timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"sensor_data_{timestamp}.csv"

# Ask for port if not found automatically
arduino_port = find_arduino_port()
if not arduino_port:
    arduino_port = input("Enter Arduino port (e.g., COM4 on Windows, /dev/ttyUSB0 on Linux): ")

try:
    print(f"Connecting to Arduino on {arduino_port}")
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(2)  # Allow connection to stabilize
    
    with open(filename, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Temperature", "Humidity", "Vibration", 
                         "AccelX", "AccelY", "AccelZ", "GyroX", "GyroY", "GyroZ"])
        
        print(f"Logging data to {filename}. Press Ctrl+C to stop.")
        print("Creating normal conditions for 1 minute, then simulate some faults...")
        print("TIP: Gently tap or shake the MPU6050 occasionally to simulate faults")
        
        count = 0
        start_time = time.time()
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                try:
                    current_time = datetime.now().strftime("%H:%M:%S")
                    data = line.split(",")
                    if len(data) == 9:  # We expect 9 values from our Arduino code
                        writer.writerow([current_time] + data)
                        file.flush()  # Ensure data is written immediately
                        
                        count += 1
                        if count % 10 == 0:  # Show status every 10 readings
                            elapsed = time.time() - start_time
                            print(f"Collected {count} readings in {elapsed:.1f} seconds")
                except Exception as e:
                    print(f"Error processing data: {e}")
                    
except KeyboardInterrupt:
    print(f"\nLogging stopped. Collected {count} readings.")
    if 'ser' in locals() and ser:
        ser.close()
except Exception as e:
    print(f"Error: {e}")
    if 'ser' in locals() and ser:
        ser.close()