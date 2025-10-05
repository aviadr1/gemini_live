let isRunning = false;
let currentGeminiMessage = null;
let currentUserTranscript = null;

async function startSession() {
    const mode = document.getElementById('video-mode').value;
    const result = await eel.start_session(mode)();

    if (result.status === 'success') {
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
        document.getElementById('stop-btn').disabled = true;
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

eel.expose(update_status);
function update_status(status) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = status;

    statusEl.className = '';
    if (status.includes('Connected')) {
        statusEl.classList.add('connected');
        isRunning = true;
        updateUIState(true);
    } else if (status.includes('Error')) {
        statusEl.classList.add('error');
        isRunning = false;
        updateUIState(false);
    } else if (status.includes('Disconnected')) {
        isRunning = false;
        updateUIState(false);
        currentGeminiMessage = null;
        currentUserTranscript = null;
    }
}

// For typed text messages
eel.expose(add_message);
function add_message(sender, text) {
    const transcript = document.getElementById('transcript');

    if (sender === 'user') {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `<strong>You:</strong> <span class="message-text">${escapeHtml(text)}</span>`;
        transcript.appendChild(messageDiv);
        currentGeminiMessage = null;
    } else if (sender === 'gemini') {
        if (!currentGeminiMessage) {
            currentGeminiMessage = document.createElement('div');
            currentGeminiMessage.className = 'message gemini-message';
            currentGeminiMessage.innerHTML = '<strong>Gemini:</strong> <span class="message-text"></span>';
            transcript.appendChild(currentGeminiMessage);
        }
        const textSpan = currentGeminiMessage.querySelector('.message-text');
        textSpan.textContent += text;
    }

    transcript.scrollTop = transcript.scrollHeight;
}

// User voice transcription (incremental)
eel.expose(add_user_transcription);
function add_user_transcription(text) {
    const transcript = document.getElementById('transcript');

    if (!currentUserTranscript) {
        currentUserTranscript = document.createElement('div');
        currentUserTranscript.className = 'message user-message voice-transcript';
        currentUserTranscript.innerHTML = '<strong>🎤 You:</strong> <span class="message-text"></span>';
        transcript.appendChild(currentUserTranscript);
    }

    const textSpan = currentUserTranscript.querySelector('.message-text');
    textSpan.textContent += text;
    transcript.scrollTop = transcript.scrollHeight;
}

// Gemini voice transcription (incremental)
eel.expose(add_gemini_transcription);
function add_gemini_transcription(text) {
    const transcript = document.getElementById('transcript');

    if (!currentGeminiMessage) {
        currentGeminiMessage = document.createElement('div');
        currentGeminiMessage.className = 'message gemini-message voice-transcript';
        currentGeminiMessage.innerHTML = '<strong>🔊 Gemini:</strong> <span class="message-text"></span>';
        transcript.appendChild(currentGeminiMessage);
    }

    const textSpan = currentGeminiMessage.querySelector('.message-text');
    textSpan.textContent += text;
    transcript.scrollTop = transcript.scrollHeight;
}

// Finalize user voice message
eel.expose(finalize_user_message);
function finalize_user_message(fullText) {
    if (currentUserTranscript) {
        const strong = currentUserTranscript.querySelector('strong');
        strong.textContent = 'You:';
        currentUserTranscript.classList.remove('voice-transcript');
        currentUserTranscript = null;
    }
}

// Finalize Gemini voice message with optional frame data
eel.expose(finalize_gemini_message);
function finalize_gemini_message(fullText, frameData = null) {
    if (currentGeminiMessage) {
        const strong = currentGeminiMessage.querySelector('strong');
        strong.textContent = 'Gemini:';
        currentGeminiMessage.classList.remove('voice-transcript');

        // Add frame thumbnail if available
        if (frameData) {
            const frameContainer = document.createElement('div');
            frameContainer.className = 'frame-container';

            const img = document.createElement('img');
            img.src = `data:${frameData.mime_type};base64,${frameData.image}`;
            img.className = 'captured-frame';
            img.alt = 'Captured frame';

            const caption = document.createElement('div');
            caption.className = 'frame-caption';
            caption.textContent = `📷 Captured ${frameData.frame_count} frame${frameData.frame_count !== 1 ? 's' : ''} (${frameData.duration}s)`;

            frameContainer.appendChild(img);
            frameContainer.appendChild(caption);
            currentGeminiMessage.appendChild(frameContainer);
        }

        currentGeminiMessage = null;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('message-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});