from flask import Flask, render_template, send_file, jsonify
import threading
import numpy as np
import pandas as pd
import io
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import time
import traceback

app = Flask(__name__)

# Global variables
latest_data = {
    "temperature": 0, "humidity": 0, "vibration": 0, 
    "accel_x": 0, "accel_y": 0, "accel_z": 0, 
    "gyro_x": 0, "gyro_y": 0, "gyro_z": 0,
    "accel_mag": 0, "gyro_mag": 0,
    "timestamp": datetime.now().strftime("%H:%M:%S"),
    "risk": 0
}
data_history = []
thread_running = False

# Create templates folder
os.makedirs('templates', exist_ok=True)

# Create simple HTML template with explicit encoding
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Predictive Maintenance Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <meta charset="utf-8">
    <style>
        body { font-family: Arial; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        .dashboard { display: flex; flex-wrap: wrap; justify-content: center; }
        .card { background: white; border-radius: 8px; margin: 10px; padding: 15px; width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .value { font-size: 24px; font-weight: bold; }
        .status { padding: 5px; border-radius: 3px; display: inline-block; margin-top: 5px; width: 80%; }
        .normal { background: #4CAF50; color: white; }
        .warning { background: #FF9800; color: white; }
        .danger { background: #F44336; color: white; }
        .chart { margin: 20px auto; width: 90%; max-width: 800px; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        img { max-width: 100%; }
        h2 { color: #444; text-align: center; }
        table { border-collapse: collapse; width: 90%; margin: 20px auto; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Arduino Predictive Maintenance Dashboard</h1>
    
    <div class="dashboard">
        <div class="card">
            <h3>Temperature</h3>
            <div class="value">{{latest.temperature}}°C</div>
            <div class="status {{temp_status}}">{{temp_status|upper}}</div>
        </div>
        
        <div class="card">
            <h3>Humidity</h3>
            <div class="value">{{latest.humidity}}%</div>
        </div>
        
        <div class="card">
            <h3>Vibration</h3>
            <div class="value">{{latest.vibration}}</div>
            <div class="status {{vib_status}}">{{vib_status|upper}}</div>
        </div>
        
        <div class="card">
            <h3>Acceleration</h3>
            <div class="value">{{latest.accel_mag}}</div>
            <div class="status {{accel_status}}">{{accel_status|upper}}</div>
        </div>
        
        <div class="card">
            <h3>Failure Risk</h3>
            <div class="value">{{latest.risk}}%</div>
            <div class="status {{risk_status}}">{{risk_status|upper}}</div>
        </div>
    </div>
    
    <h2>Live Chart</h2>
    <div class="chart">
        <img src="/plot" style="width: 100%;">
    </div>
    
    <h2>Recent Readings</h2>
    <table>
        <tr>
            <th>Time</th>
            <th>Temp (°C)</th>
            <th>Humidity (%)</th>
            <th>Vibration</th>
            <th>Accel</th>
            <th>Risk (%)</th>
        </tr>
        {% for item in history %}
        <tr>
            <td>{{item.timestamp}}</td>
            <td>{{item.temperature}}</td>
            <td>{{item.humidity}}</td>
            <td>{{item.vibration}}</td>
            <td>{{item.accel_mag}}</td>
            <td>{{item.risk}}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
    ''')

# Simplified simulation function
def generate_simulated_data():
    """Generate simulated sensor data"""
    temp = 25 + np.random.normal(0, 3)
    humidity = 50 + np.random.normal(0, 8)
    vibration = np.random.choice([0, 0, 0, 0, 1])  # 20% chance of vibration
    
    accel_x = np.random.normal(0, 1)
    accel_y = np.random.normal(0, 1)
    accel_z = 9.8 + np.random.normal(0, 0.5)  # Gravity + noise
    
    gyro_x = np.random.normal(0, 0.1)
    gyro_y = np.random.normal(0, 0.1)
    gyro_z = np.random.normal(0, 0.1)
    
    # Occasionally simulate high values
    if np.random.random() < 0.05:  # 5% chance
        temp += 5
        accel_x *= 1.5
        accel_y *= 1.5
        accel_z *= 1.5
        vibration = 1
    
    # Calculate magnitudes
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    
    # Basic risk calculation (simplified)
    risk = 0
    if accel_mag > 10:
        risk = int((accel_mag - 10) * 10)
    if temp > 30:
        risk = max(risk, int((temp - 30) * 5))
    risk = min(100, risk)  # Cap at 100%
    
    return {
        "temperature": round(temp, 1),
        "humidity": round(humidity, 1),
        "vibration": vibration,
        "accel_x": round(accel_x, 2),
        "accel_y": round(accel_y, 2),
        "accel_z": round(accel_z, 2),
        "gyro_x": round(gyro_x, 2),
        "gyro_y": round(gyro_y, 2),
        "gyro_z": round(gyro_z, 2),
        "accel_mag": round(accel_mag, 2),
        "gyro_mag": round(gyro_mag, 2),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "risk": risk
    }

def data_thread():
    """Thread function to generate simulated data."""
    global thread_running, latest_data, data_history
    
    print("Data generation thread started")
    while thread_running:
        try:
            # Generate new data
            new_data = generate_simulated_data()
            
            # Update global data
            latest_data = new_data
            
            # Add to history (limited to last 20 entries)
            data_history.append(new_data.copy())
            if len(data_history) > 20:
                data_history.pop(0)
                
            # Sleep to control data generation rate
            time.sleep(1)
        except Exception as e:
            print(f"Error in data thread: {e}")
            traceback.print_exc()
            time.sleep(1)  # Prevent error loop

@app.route('/')
def index():
    """Main dashboard page."""
    global latest_data, data_history
    
    try:
        # Determine status for each metric
        temp_status = "normal"
        if latest_data["temperature"] > 30:
            temp_status = "warning"
        if latest_data["temperature"] > 35:
            temp_status = "danger"
            
        vib_status = "normal"
        if latest_data["vibration"] == 1:
            vib_status = "warning"
            
        accel_status = "normal"
        if latest_data["accel_mag"] > 10:
            accel_status = "warning"
        if latest_data["accel_mag"] > 14:
            accel_status = "danger"
            
        risk_status = "normal"
        if latest_data["risk"] > 30:
            risk_status = "warning"
        if latest_data["risk"] > 60:
            risk_status = "danger"
        
        return render_template('index.html', 
                              latest=latest_data,
                              history=list(reversed(data_history)),
                              temp_status=temp_status,
                              vib_status=vib_status,
                              accel_status=accel_status,
                              risk_status=risk_status)
    except Exception as e:
        print(f"Error in index route: {e}")
        traceback.print_exc()
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/plot')
def plot():
    """Generate plot for dashboard."""
    global data_history
    
    try:
        if not data_history:
            # Create empty plot if no data
            plt.figure(figsize=(10, 6))
            plt.title("No data available")
            plt.grid(True)
            
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()
            return send_file(img, mimetype='image/png')
        
        # Create DataFrame from history
        df = pd.DataFrame(data_history)
        
        # Get timestamps
        x = list(range(len(df)))
        
        plt.figure(figsize=(10, 6))
        
        # Plot temperature and risk
        plt.subplot(2, 1, 1)
        plt.plot(x, df["temperature"], 'r-', label="Temperature (°C)")
        plt.fill_between(x, 0, df["risk"], color='purple', alpha=0.3, label="Risk %")
        plt.legend()
        plt.grid(True)
        plt.title("Temperature and Failure Risk")
        
        # Plot acceleration and vibration
        plt.subplot(2, 1, 2)
        plt.plot(x, df["accel_mag"], 'g-', label="Accel Magnitude")
        plt.bar(x, df["vibration"], color='orange', alpha=0.7, label="Vibration")
        plt.legend()
        plt.grid(True)
        plt.title("Acceleration and Vibration")
        
        plt.tight_layout()
        
        # Save to bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        return send_file(img, mimetype='image/png')
    except Exception as e:
        print(f"Plot error: {e}")
        traceback.print_exc()
        
        # Return a simple error plot
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return send_file(img, mimetype='image/png')

@app.route('/data')
def get_data():
    """API endpoint to get current sensor data."""
    global latest_data
    return jsonify(latest_data)

@app.route('/debug')
def debug():
    """Debug endpoint to check if server is running."""
    return "Flask server is running. Basic routing works."

def start_server():
    """Initialize and start the Flask server."""
    global thread_running
    
    # Start data collection thread
    thread_running = True
    sim_thread = threading.Thread(target=data_thread)
    sim_thread.daemon = True
    sim_thread.start()
    
    try:
        # Start Flask server
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        print(f"Server error: {e}")
        traceback.print_exc()
    finally:
        # Cleanup when server stops
        thread_running = False
        if sim_thread.is_alive():
            sim_thread.join(timeout=1)

if __name__ == "__main__":
    print("Starting Predictive Maintenance Dashboard")
    print("Access at http://localhost:5000")
    print("Press Ctrl+C to stop")
    start_server()