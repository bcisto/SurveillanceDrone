document.addEventListener('DOMContentLoaded', () => {
    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speed-value');
    const status = document.getElementById('status');
    let currentSpeed = 50;
    let isMoving = false;
    let currentDirection = 0;

    // Sensor elements
    const sensorElements = {
        // System metrics
        batteryLevel: document.getElementById('battery-level'),
        cpuTemp: document.getElementById('cpu-temp'),
        currentSpeed: document.getElementById('current-speed'),
        totalDistance: document.getElementById('total-distance')
    };

    // Function to update sensor displays
    function updateSensorDisplays(data) {
        if (data.system) {
            // Battery level
            const battery = data.system.battery;
            sensorElements.batteryLevel.textContent = `${battery.toFixed(1)}%`;
            
            // Update battery icon color based on level
            const batteryIcon = sensorElements.batteryLevel.previousElementSibling;
            if (battery < 20) {
                batteryIcon.style.color = '#f44336'; // Red for low battery
                sensorElements.batteryLevel.style.color = '#f44336';
            } else if (battery < 50) {
                batteryIcon.style.color = '#ff9800'; // Orange for medium battery
                sensorElements.batteryLevel.style.color = '#ff9800';
            } else {
                batteryIcon.style.color = '#4caf50'; // Green for good battery
                sensorElements.batteryLevel.style.color = '#4caf50';
            }
            
            // Also update the battery icon data
            batteryIcon.setAttribute('data-icon', `mdi:battery${battery >= 90 ? '' : battery >= 70 ? '-80' : battery >= 50 ? '-60' : battery >= 30 ? '-40' : battery >= 10 ? '-20' : '-outline'}`);
            
            // Temperature (implemented)
            const temp = data.system.temperature;
            sensorElements.cpuTemp.textContent = `${temp.toFixed(1)}°C`;
            
            // Update temperature icon color based on value
            const tempIcon = sensorElements.cpuTemp.previousElementSibling;
            if (temp > 80) {
                tempIcon.style.color = '#f44336'; // Red for high temp
                sensorElements.cpuTemp.style.color = '#f44336';
            } else if (temp > 70) {
                tempIcon.style.color = '#ff9800'; // Orange for medium temp
                sensorElements.cpuTemp.style.color = '#ff9800';
            } else {
                tempIcon.style.color = '#4caf50'; // Green for good temp
                sensorElements.cpuTemp.style.color = '#4caf50';
            }
            
            // Speed and distance (implemented)
            sensorElements.currentSpeed.textContent = `${(data.system.speed).toFixed(2)} m/s`;
            sensorElements.totalDistance.textContent = `${data.system.total_distance.toFixed(1)} m`;
        }
    }

    // Function to fetch sensor data
    async function fetchSensorData() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            if (data.status === 'success') {
                updateSensorDisplays(data);
            }
        } catch (error) {
            console.error('Error fetching sensor data:', error);
        }
    }

    // Update speed value display
    speedSlider.addEventListener('input', (e) => {
        currentSpeed = parseInt(e.target.value);
        speedValue.textContent = currentSpeed;
        
        // If we're currently moving, update the speed
        if (isMoving) {
            sendCommand(isMoving === 'forward' ? 'forward' : 'backward');
        }
    });

    // Function to send commands to the server
    async function sendCommand(command, value = null) {
        try {
            const response = await fetch('/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    command: command,
                    value: value || currentSpeed
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Update our state based on server response
                if (data.state) {
                    isMoving = data.state.moving ? (data.state.speed > 0 ? 'forward' : 'backward') : false;
                    currentDirection = data.state.direction;
                }
                
                status.textContent = `Command: ${command} | Speed: ${currentSpeed} | Direction: ${currentDirection}°`;
                status.style.background = '#e8f5e9';
                status.style.color = '#2e7d32';
            } else {
                status.textContent = `Error: ${data.message}`;
                status.style.background = '#ffebee';
                status.style.color = '#c62828';
                // On error, reset our state
                isMoving = false;
            }
        } catch (error) {
            status.textContent = `Error: ${error.message}`;
            status.style.background = '#ffebee';
            status.style.color = '#c62828';
            isMoving = false;
        }
    }

    // Control button event listeners
    document.getElementById('forward').addEventListener('mousedown', () => {
        isMoving = 'forward';
        sendCommand('forward');
    });

    document.getElementById('backward').addEventListener('mousedown', () => {
        isMoving = 'backward';
        sendCommand('backward');
    });

    // Stop on mouse up for movement buttons
    ['forward', 'backward'].forEach(id => {
        document.getElementById(id).addEventListener('mouseup', () => {
            isMoving = false;
            sendCommand('stop');
        });
        
        // Also stop if mouse leaves the button while held down
        document.getElementById(id).addEventListener('mouseleave', () => {
            if (isMoving) {
                isMoving = false;
                sendCommand('stop');
            }
        });
    });

    document.getElementById('left').addEventListener('click', () => {
        currentDirection = -30;
        sendCommand('turn_left', 30);
    });

    document.getElementById('right').addEventListener('click', () => {
        currentDirection = 30;
        sendCommand('turn_right', 30);
    });

    document.getElementById('stop').addEventListener('click', () => {
        isMoving = false;
        sendCommand('stop');
    });

    // Keyboard controls with continuous movement
    const keyState = {};
    
    window.addEventListener('keydown', (e) => {
        if (keyState[e.key]) return; // Prevent key repeat
        keyState[e.key] = true;
        
        switch(e.key) {
            case 'ArrowUp':
                isMoving = 'forward';
                sendCommand('forward');
                break;
            case 'ArrowDown':
                isMoving = 'backward';
                sendCommand('backward');
                break;
            case 'ArrowLeft':
                currentDirection = -30;
                sendCommand('turn_left', 30);
                break;
            case 'ArrowRight':
                currentDirection = 30;
                sendCommand('turn_right', 30);
                break;
            case ' ': // Spacebar
                isMoving = false;
                sendCommand('stop');
                break;
        }
    });

    window.addEventListener('keyup', (e) => {
        keyState[e.key] = false;
        
        switch(e.key) {
            case 'ArrowUp':
                if (isMoving === 'forward') {
                    isMoving = false;
                    sendCommand('stop');
                }
                break;
            case 'ArrowDown':
                if (isMoving === 'backward') {
                    isMoving = false;
                    sendCommand('stop');
                }
                break;
        }
    });

    // Handle page visibility change
    document.addEventListener('visibilitychange', () => {
        if (document.hidden && isMoving) {
            isMoving = false;
            sendCommand('stop');
        }
    });

    // Start periodic sensor updates
    fetchSensorData(); // Initial fetch
    setInterval(fetchSensorData, 200); // Update every 200ms
}); 