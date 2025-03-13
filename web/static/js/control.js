document.addEventListener('DOMContentLoaded', () => {
    const speedSlider = document.getElementById('speed');
    const speedValue = document.getElementById('speed-value');
    const status = document.getElementById('status');
    let currentSpeed = 50;
    let isMoving = false;
    let currentDirection = 0;

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
}); 