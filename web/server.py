from flask import Flask, render_template, jsonify, request
from picarx import Picarx
from time import sleep

app = Flask(__name__)

# Store the current state
current_state = {
    'speed': 0,
    'direction': 0,
    'moving': False
}

@app.route('/')
def index():
    return render_template('index.html')

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
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        # Proper cleanup like in keyboard example
        px.set_dir_servo_angle(0)
        px.stop()
        sleep(0.2) 