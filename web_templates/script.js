let isRunning = false;

async function startSession() {
    const mode = document.getElementById('video-mode').value;
    const result = await eel.start_session(mode)();

    if (result.status === 'success') {
        // Wait a moment for the session to actually start
        setTimeout(() => {
            isRunning = true;
            updateUIState(true);
        }, 600);
    } else {
        alert('Error: ' + result.message);
    }
}

async function stopSession() {
    const result = await eel.stop_session()();

    if (result.status === 'success') {
        // Don't immediately change UI state - wait for the status update from Python
        // The Python side will call update_status("Disconnected") when actually stopped
        document.getElementById('stop-btn').disabled = true; // Just disable the button to prevent multiple clicks
    } else {
        alert('Error: ' + result.message);
    }
}

async function sendMessage() {
    const input = document.getElementById('message-input');
    const text = input.value.trim();

    if (!text) return;

    const result = await eel.send_message(text)();

    if (result.status === 'success') {
        input.value = '';
    } else {
        alert('Error sending message: ' + result.message);
    }
}

function updateUIState(running) {
    document.getElementById('start-btn').disabled = running;
    document.getElementById('stop-btn').disabled = !running;
    document.getElementById('message-input').disabled = !running;
    document.getElementById('send-btn').disabled = !running;
    document.getElementById('video-mode').disabled = running;
}

// Exposed function for Python to call
eel.expose(update_status);
function update_status(status) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = status;

    statusEl.className = '';
    if (status.includes('Connected')) {
        statusEl.classList.add('connected');
        isRunning = true;
        updateUIState(true);  // THIS IS MISSING
    } else if (status.includes('Error')) {
        statusEl.classList.add('error');
        isRunning = false;
        updateUIState(false);  // THIS IS MISSING
    } else if (status.includes('Disconnected')) {  // THIS WHOLE BLOCK IS MISSING
        // Session has actually stopped
        isRunning = false;
        updateUIState(false);
    }
}

// Exposed function for Python to call
eel.expose(update_transcript);
function update_transcript(text) {
    const transcript = document.getElementById('transcript');
    transcript.textContent += text;
    // Auto-scroll to bottom
    transcript.scrollTop = transcript.scrollHeight;
}

// Allow Enter key to send message
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('message-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});