from flask import Flask, render_template, jsonify, request
from picarx import Picarx
from time import sleep, time
import psutil  # For CPU temp and battery
import os

# Import Robot HAT for direct ADC access
try:
    from robot_hat import ADC
    print("Successfully imported robot_hat.ADC")
except Exception as e:
    print(f"Error importing robot_hat.ADC: {e}")

app = Flask(__name__)

# Store the current state and metrics
current_state = {
    'speed': 0,
    'direction': 0,
    'moving': False,
    'total_distance': 0,  # in meters
    'last_update_time': time()  # Initialize with current time
}

# Initialize ADC for battery reading
try:
    # Battery voltage is connected to A4 through a voltage divider
    battery_adc = ADC('A4')
    voltage = battery_adc.read_voltage()
    print(f"Battery ADC A4 voltage reading: {voltage:.2f}V")
    print(f"Calculated battery voltage: {voltage * 3:.2f}V")
    
    if not battery_adc:
        print("Warning: Failed to initialize battery ADC")
except Exception as e:
    print(f"Error initializing ADC: {e}")
    battery_adc = None

def get_cpu_temperature():
    try:
        # Try reading from thermal zone first (more reliable)
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = float(f.read().strip()) / 1000
        return temp
    except:
        try:
            # Fallback to vcgencmd
            temp = os.popen("vcgencmd measure_temp").readline()
            return float(temp.replace("temp=", "").replace("'C\n", ""))
        except:
            return 0

def get_battery_level():
    try:
        if not battery_adc:
            print("ADC not initialized")
            return 0
            
        # Read voltage from A4
        adc_voltage = battery_adc.read_voltage()
        # Calculate actual battery voltage (voltage divider ratio 20K/10K)
        battery_voltage = adc_voltage * 3
        
        print(f"ADC voltage: {adc_voltage:.2f}V, Battery voltage: {battery_voltage:.2f}V")
        
        # Convert voltage to percentage
        # For a 2S Li-Po battery:
        # 8.4V = 100% (fully charged)
        # 6.0V = 0% (cutoff voltage)
        MAX_VOLTAGE = 8.4
        MIN_VOLTAGE = 6.0
        voltage_range = MAX_VOLTAGE - MIN_VOLTAGE
        
        # Calculate percentage
        percentage = ((battery_voltage - MIN_VOLTAGE) / voltage_range) * 100
        
        # Clamp between 0 and 100
        percentage = max(0, min(100, percentage))
        
        print(f"Calculated battery percentage: {percentage:.1f}%")
        return percentage
    except Exception as e:
        print(f"Error reading battery: {type(e).__name__}: {e}")
        return 0

def update_distance_and_speed(current_speed):
    """Update total distance based on speed and time elapsed"""
    current_time = time()
    time_elapsed = current_time - current_state['last_update_time']
    
    # Convert speed percentage to approximate m/s (assuming max speed is about 1 m/s)
    speed_ms = (current_speed / 100.0) * 1.0
    
    # Update total distance only if moving
    if current_state['moving']:
        distance_traveled = speed_ms * time_elapsed
        current_state['total_distance'] += distance_traveled
    
    current_state['last_update_time'] = current_time
    return speed_ms

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    try:
        # Get system metrics
        cpu_temp = get_cpu_temperature()
        battery = get_battery_level()
        print(f"Status update - Battery: {battery}%, CPU Temp: {cpu_temp}°C")
        
        # Update distance and get current speed in m/s
        current_speed = update_distance_and_speed(abs(current_state['speed']))
        
        return jsonify({
            'status': 'success',
            'system': {
                'temperature': cpu_temp,
                'battery': battery,
                'speed': current_speed,
                'total_distance': current_state['total_distance']
            },
            'movement': current_state
        })
    except Exception as e:
        print(f"Error reading sensor data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/control', methods=['POST'])
def control():
    command = request.json.get('command')
    value = float(request.json.get('value', 0))  # Convert to float for servo angles
    
    try:
        if command == 'forward':
            current_state['speed'] = value
            current_state['moving'] = True
            px.set_dir_servo_angle(0)  # Reset direction before moving
            px.forward(value)
        elif command == 'backward':
            current_state['speed'] = -value
            current_state['moving'] = True
            px.set_dir_servo_angle(0)  # Reset direction before moving
            px.backward(value)
        elif command == 'turn_left':
            current_state['direction'] = -value
            px.set_dir_servo_angle(-30)  # Fixed angle like in keyboard example
            px.forward(80)  # Fixed speed like in keyboard example
        elif command == 'turn_right':
            current_state['direction'] = value
            px.set_dir_servo_angle(30)  # Fixed angle like in keyboard example
            px.forward(80)  # Fixed speed like in keyboard example
        elif command == 'stop':
            current_state['speed'] = 0
            current_state['moving'] = False
            px.forward(0)  # Stop moving but maintain direction
            sleep(0.2)  # Consistent delay from example
        
        return jsonify({
            'status': 'success',
            'state': current_state
        })
    except Exception as e:
        print(f"Error executing command {command}: {e}")  # Added debug print
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    try:
        px = Picarx()
        print("PicarX initialized successfully")
        
        # Debug: Check available methods
        print("\nChecking available battery reading methods:")
        print("get_battery_voltage:", hasattr(px, 'get_battery_voltage'))
        print("get_power_voltage:", hasattr(px, 'get_power_voltage'))
        print("get_voltage:", hasattr(px, 'get_voltage'))
        if hasattr(px, 'adc'):
            print("ADC object exists")
            print("adc.get_battery_voltage:", hasattr(px.adc, 'get_battery_voltage'))
        
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Proper cleanup like in keyboard example
        px.set_dir_servo_angle(0)
        px.stop()
        sleep(0.2) 