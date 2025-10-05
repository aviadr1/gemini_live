let isRunning = false;
let currentGeminiMessage = null;
let currentUserTranscript = null;
let systemPrompts = {};

// Load system prompts when page loads
async function loadSystemPrompts() {
    systemPrompts = await eel.get_system_prompts()();
}

// Update prompt display when selection changes
function updatePromptDisplay() {
    const select = document.getElementById('system-prompt-select');
    const selectedKey = select.value;

    const instructionDisplay = document.getElementById('system-instruction-display');
    const instructionText = document.getElementById('system-instruction-text');
    const promptsDisplay = document.getElementById('narration-prompts-display');
    const promptsList = document.getElementById('narration-prompts-list');
    const customSection = document.getElementById('custom-prompt-section');

    if (selectedKey === 'custom') {
        // Show custom inputs
        instructionDisplay.style.display = 'none';
        promptsDisplay.style.display = 'none';
        customSection.style.display = 'block';
    } else if (selectedKey && systemPrompts[selectedKey]) {
        // Show predefined prompt
        const prompt = systemPrompts[selectedKey];

        instructionText.textContent = prompt.system_instruction;
        instructionDisplay.style.display = 'block';

        promptsList.innerHTML = '';
        prompt.narration_prompts.forEach(p => {
            const li = document.createElement('li');
            li.textContent = p;
            promptsList.appendChild(li);
        });
        promptsDisplay.style.display = 'block';

        customSection.style.display = 'none';
    } else {
        // Hide all
        instructionDisplay.style.display = 'none';
        promptsDisplay.style.display = 'none';
        customSection.style.display = 'none';
    }
}

// Update FPS display when slider changes
document.addEventListener('DOMContentLoaded', async function() {
    await loadSystemPrompts();

    const fpsSlider = document.getElementById('fps-slider');
    const fpsValue = document.getElementById('fps-value');
    const narrationMode = document.getElementById('narration-mode');
    const intervalGroup = document.getElementById('narration-interval-group');
    const intervalSlider = document.getElementById('interval-slider');
    const intervalValue = document.getElementById('interval-value');
    const messageInput = document.getElementById('message-input');
    const systemPromptSelect = document.getElementById('system-prompt-select');

    // FPS slider
    fpsSlider.addEventListener('input', function() {
        fpsValue.textContent = this.value;
    });

    // Narration mode toggle
    narrationMode.addEventListener('change', function() {
        intervalGroup.style.display = this.checked ? 'flex' : 'none';
    });

    // Interval slider
    intervalSlider.addEventListener('input', function() {
        intervalValue.textContent = this.value;
    });

    // System prompt selection
    systemPromptSelect.addEventListener('change', updatePromptDisplay);

    // Enter key to send message
    messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});

async function startSession() {
    const mode = document.getElementById('video-mode').value;
    const fps = parseFloat(document.getElementById('fps-slider').value);
    const narrationMode = document.getElementById('narration-mode').checked;
    const narrationInterval = parseInt(document.getElementById('interval-slider').value);
    const systemPromptKey = document.getElementById('system-prompt-select').value;

    let customSystemPrompt = '';
    let customNarrationPrompts = '';

    if (systemPromptKey === 'custom') {
        customSystemPrompt = document.getElementById('custom-system-instruction').value;
        customNarrationPrompts = document.getElementById('custom-narration-prompts').value;

        if (!customSystemPrompt) {
            alert('Please enter a custom system instruction or select a predefined prompt.');
            return;
        }
    }

    const result = await eel.start_session(
        mode,
        fps,
        narrationMode,
        narrationInterval,
        systemPromptKey === 'custom' ? '' : systemPromptKey,
        customSystemPrompt,
        customNarrationPrompts
    )();

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
    document.getElementById('fps-slider').disabled = running;
    document.getElementById('narration-mode').disabled = running;
    document.getElementById('interval-slider').disabled = running;
    document.getElementById('system-prompt-select').disabled = running;
    document.getElementById('custom-system-instruction').disabled = running;
    document.getElementById('custom-narration-prompts').disabled = running;
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