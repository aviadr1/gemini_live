# Gemini Live API - Real-time Multimodal Interaction

A Python application for real-time bidirectional communication with Google's Gemini AI using voice, video, and text. Features both a command-line interface and a web-based UI with configurable video frame rates, automatic transcription, and AI-powered video narration.

## Features

- **Real-time Voice Conversation**: Speak naturally with Gemini using your microphone and speakers
- **Video Input**: Share your camera or screen with Gemini for visual context
- **Automatic Transcription**: See real-time transcriptions of both your speech and Gemini's responses
- **Text Input**: Type messages in addition to voice
- **Interruption Support**: Interrupt Gemini mid-response by speaking
- **Web UI**: Modern browser-based interface with message history and video preview
- **Console Mode**: Lightweight command-line interface
- **Configurable Frame Rate**: Adjust video capture from 0.5 to 5 FPS
- **Video Narration Mode**: Get continuous AI feedback on video content at customizable intervals
- **Predefined System Prompts**: Choose from curated scenarios like Video Describer, Facial Expression Analyzer, and Yoga Instructor

## Requirements

- Python 3.12+
- Google AI API key
- Microphone and speakers (headphones strongly recommended)
- Optional: Webcam for camera mode

You're right! Let me update the installation section:

## Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies using `uv`:
```bash
uv sync
```

Or using Poetry:
```bash
poetry install
```

3. Create a `.env` file with your API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Usage

### Web UI (Recommended)

Launch the web interface:
```bash
uv run python app_ui.py
# Or with Poetry:
poetry run python app_ui.py
```

### Console Mode

For a lightweight command-line experience:

```bash
# Camera mode (default)
uv run python main.py

# Screen share mode
uv run python main.py --mode screen

# Audio only mode
uv run python main.py --mode none
```

## Usage

### Web UI (Recommended)

Launch the web interface:
```bash
python app_ui.py
```

The browser will open automatically. Features include:
- Visual message history with timestamps
- Video preview showing what Gemini sees
- Configurable settings (FPS, narration mode, system prompts)
- Real-time transcription display
- Easy session control

**Configuration Options:**
- **Video Mode**: Camera, Screen Share, or None
- **Frame Rate**: 0.5 to 5 FPS (default: 1 FPS)
- **Narration Mode**: Enable periodic AI commentary on video
- **Narration Interval**: Seconds between narration prompts (default: 5s)
- **System Prompts**: Choose predefined scenarios or write custom instructions

### Console Mode

For a lightweight command-line experience:

```bash
# Camera mode (default)
python main.py

# Screen share mode
python main.py --mode screen

# Audio only mode
python main.py --mode none
```

Type messages at the `message >` prompt or speak naturally. Type `q` to quit.

## System Prompts

The web UI includes three predefined system prompts:

1. **Video Describer for Accessibility**: Detailed descriptions of visual content for blind/low-vision users
2. **Facial Expression Analyzer**: Analyzes facial expressions and emotional states
3. **Yoga Instructor**: Provides form feedback and guidance for yoga practice

You can also write custom system instructions and narration prompts.

## How It Works

### Architecture

The application uses 6 concurrent tasks:
1. **Text Input**: Accepts keyboard input
2. **Audio Capture**: Records from microphone (16kHz PCM)
3. **Video Capture**: Captures camera/screen at configured FPS
4. **Data Sender**: Streams audio/video to Gemini
5. **Response Receiver**: Receives Gemini's responses
6. **Audio Playback**: Plays Gemini's voice (24kHz PCM)

### Key Concepts

- **Session**: Persistent WebSocket connection maintaining conversation state
- **Turn**: Single exchange of user input and model response
- **VAD**: Automatic Voice Activity Detection for turn boundaries
- **Interruption**: Speak anytime to interrupt Gemini's response
- **Streaming**: Real-time data transmission without explicit turn boundaries

## Important Notes

### Audio Setup

**Use headphones!** The system uses default audio input/output without echo cancellation. Without headphones, Gemini may hear itself and interrupt its own responses.

### Video Processing

- Default frame rate: 1 FPS (balances quality and performance)
- Higher FPS (2-5): More responsive but uses more bandwidth
- Lower FPS (0.5): Reduces costs and bandwidth usage
- Narration mode: AI provides periodic commentary on video content

### Transcription

All audio is automatically transcribed:
- Your speech appears as `[You: transcript]`
- Gemini's speech appears inline with audio playback
- Web UI shows transcriptions in message bubbles with timestamps

## Project Structure

```
.
├── app_ui.py              # Web UI application with Eel
├── main.py                # Console application
├── utils.py               # Shared utilities (camera/screen capture)
├── web_templates/         # HTML/CSS/JS templates
│   ├── index.html
│   ├── style.css
│   └── script.js
├── web/                   # Generated web files (auto-synced)
└── .env                   # API key (create this)
```

## Troubleshooting

**No audio input/output**: Check your system's default audio devices and ensure PyAudio is properly installed.

**Video not capturing**: Verify camera permissions and that no other application is using the camera.

**API errors**: Confirm your `GOOGLE_API_KEY` is valid and has access to the Gemini Live API.

**Echo/feedback**: Use headphones to prevent the microphone from picking up speaker output.

**High latency**: Try reducing the frame rate or switching to audio-only mode.

## Debug Mode

Enable detailed logging by keeping `DEBUG = True` in `app_ui.py`. This shows:
- Frame capture and transmission counts
- Audio chunk statistics
- Task lifecycle events
- Error traces

## License

Apache License 2.0 - See code headers for full license text.

## Credits

Built with:
- [Google Gemini API](https://ai.google.dev/)
- [Eel](https://github.com/python-eel/Eel) for web UI
- OpenCV for video capture
- PyAudio for audio I/O