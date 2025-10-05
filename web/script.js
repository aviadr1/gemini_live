let isRunning = false;

async function startSession() {
    const mode = document.getElementById('video-mode').value;
    const result = await eel.start_session(mode)();

    if (result.status === 'success') {
        isRunning = true;
        updateUIState(true);
    } else {
        alert('Error: ' + result.message);
    }
}

async function stopSession() {
    const result = await eel.stop_session()();
    isRunning = false;
    updateUIState(false);
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
    } else if (status.includes('Error')) {
        statusEl.classList.add('error');
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