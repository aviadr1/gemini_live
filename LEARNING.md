# LEARNING.md - Understanding Gemini Live API

A comprehensive guide to understanding the Gemini Live API and the architecture of this real-time multimodal application.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [How Gemini Live API Works](#how-gemini-live-api-works)
3. [Key Architectural Patterns](#key-architectural-patterns)
4. [Critical Code Sections in app_ui.py](#critical-code-sections-in-appuipy)
5. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
6. [Best Practices](#best-practices)

---

## Core Concepts

### What is the Live API?

The Gemini Live API enables **bidirectional streaming** communication with Google's Gemini AI model. Unlike traditional request-response APIs, Live API maintains a persistent WebSocket connection that allows:

- **Continuous audio streaming** (both directions)
- **Real-time video/image transmission** to the model
- **Automatic transcription** of all audio
- **Voice Activity Detection (VAD)** to detect when you start/stop speaking
- **Interruption handling** - you can interrupt the model mid-response

### Key Terms

- **Session**: A WebSocket connection that maintains conversation context
- **Turn**: A complete interaction cycle (user input → model response)
- **Realtime Input**: Streaming data (audio/video) sent without explicit turn boundaries
- **Client Content**: Structured messages with explicit turn boundaries (like typed text)
- **VAD (Voice Activity Detection)**: Automatic detection of speech start/stop
- **Modality**: Input/output format (AUDIO, TEXT, or both)

---

## How Gemini Live API Works

### Connection Flow

```python
# 1. Create a client
client = genai.Client(http_options={"api_version": "v1beta"})

# 2. Configure the session
config = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],  # How the model responds
    input_audio_transcription=types.AudioTranscriptionConfig(),  # Transcribe your voice
    output_audio_transcription=types.AudioTranscriptionConfig()  # Transcribe model's voice
)

# 3. Connect (establishes WebSocket)
async with client.aio.live.connect(model=MODEL, config=config) as session:
    # Session is now active - send/receive data
    pass
```

### Two Ways to Send Data

#### 1. Realtime Input (Streaming)
Used for continuous audio/video:

```python
# Send audio chunk
await session.send_realtime_input(
    audio=types.Blob(data=audio_bytes, mime_type="audio/pcm")
)

# Send video frame
await session.send_realtime_input(
    media=types.Blob(data=image_bytes, mime_type="image/jpeg")
)
```

**Characteristics**:
- No explicit turn boundaries
- Relies on VAD to detect when you finish speaking
- Low latency
- Can be sent while model is generating

#### 2. Client Content (Structured)
Used for typed messages:

```python
await session.send_client_content(
    turns=types.Content(
        role="user",
        parts=[types.Part(text="Hello Gemini")]
    ),
    turn_complete=True  # Signals: "I'm done, please respond"
)
```

**Characteristics**:
- Explicit turn boundaries
- Immediately triggers a response when `turn_complete=True`
- Used for text messages and narration prompts

### Receiving Responses

Responses arrive as an async iterator of turns:

```python
turn = session.receive()  # Get next turn

async for response in turn:
    # Audio data to play
    if response.data:
        audio_queue.put(response.data)
    
    # Text response (if TEXT modality enabled)
    if response.text:
        print(response.text)
    
    # Your speech transcribed
    if response.server_content.input_transcription:
        print(f"You said: {response.server_content.input_transcription.text}")
    
    # Model's speech transcribed
    if response.server_content.output_transcription:
        print(f"Gemini says: {response.server_content.output_transcription.text}")
```

---

## Key Architectural Patterns

### Concurrent Task Architecture

This application uses **6 concurrent tasks** running simultaneously:

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(listen_audio())      # Capture microphone
    tg.create_task(capture_camera())    # Capture video frames
    tg.create_task(send_realtime())     # Send audio/video to API
    tg.create_task(receive_audio())     # Receive responses
    tg.create_task(play_audio())        # Play audio through speakers
    tg.create_task(wait_for_stop())     # Wait for stop signal
```

### Queue-Based Communication

Tasks communicate via **asyncio.Queue**:

```
Microphone → out_queue → Sender → [API] → Receiver → audio_in_queue → Speaker
Camera    ↗                                                           ↘
```

This decouples capture/playback from network I/O, preventing blocking.

### Synchronous Operations in Async Context

Audio/video capture are **synchronous blocking operations**. We use `asyncio.to_thread()` to prevent blocking the event loop:

```python
# WRONG - blocks entire event loop
data = audio_stream.read(CHUNK_SIZE)

# RIGHT - runs in thread pool
data = await asyncio.to_thread(audio_stream.read, CHUNK_SIZE)
```

---

## Critical Code Sections in app_ui.py

### 1. Session Initialization (Lines ~680-730)

**What it does**: Creates the Live API connection with custom system instructions.

```python
def create_live_config(system_instruction: Optional[str] = None) -> LiveConnectConfig:
    config_dict = {
        "response_modalities": [Modality.AUDIO],  # Voice responses
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "realtime_input_config": types.RealtimeInputConfig(
            turn_coverage=TurnCoverage.TURN_INCLUDES_ALL_INPUT
        )
    }
    
    if system_instruction:
        config_dict["system_instruction"] = system_instruction
    
    return types.LiveConnectConfig(**config_dict)
```

**Key insight**: System instructions shape the model's behavior. The `turn_coverage` parameter determines which input data the model considers when generating responses.

### 2. Frame Capture Loop (Lines ~240-280)

**What it does**: Captures video frames at configured FPS and queues them.

```python
async def capture_camera(self) -> None:
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
    
    while not self.should_stop:
        # Capture frame (blocking operation - use thread)
        frame = await asyncio.to_thread(capture_camera_frame, cap)
        
        if frame is None:
            break
        
        self.latest_frame = frame
        self.turn_frames.append(frame)  # Track for this turn
        
        # Wait based on FPS setting
        await asyncio.sleep(self.frame_interval)
        
        # Queue for sending
        await self.out_queue.put(frame)
    
    cap.release()
```

**Key insights**:
- `self.frame_interval = 1.0 / fps` converts FPS to seconds between frames
- `turn_frames` tracks all frames sent during current turn
- Blocking operations (capture, VideoCapture) must use `asyncio.to_thread()`

### 3. Data Sender (Lines ~330-360)

**What it does**: Dequeues data and sends to API via WebSocket.

```python
async def send_realtime(self) -> None:
    while not self.should_stop:
        msg = await self.out_queue.get()
        mime_type = msg.get("mime_type", "")
        
        if "audio" in mime_type:
            # Send audio chunk
            await self.session.send_realtime_input(
                audio=types.Blob(data=msg["data"], mime_type=mime_type)
            )
        else:
            # Send video frame (decode base64 first)
            image_data = base64.b64decode(msg["data"])
            
            self.turn_frames.append(msg)  # Track frame
            self.frames_sent_since_narration += 1
            
            await self.session.send_realtime_input(
                media=types.Blob(data=image_data, mime_type=mime_type)
            )
```

**Key insight**: All media (images/video) must be decoded from base64 back to bytes before creating a `Blob`. Audio is already in bytes format (raw PCM).

### 4. Response Receiver (Lines ~370-490)

**What it does**: Receives turns from API and processes responses.

```python
async def receive_audio(self) -> None:
    while not self.should_stop:
        turn = self.session.receive()  # Get next turn
        
        self.current_gemini_transcript = []
        self.turn_frames = []  # Reset for new turn
        
        async for response in turn:
            # Audio to play
            if data := response.data:
                self.audio_in_queue.put_nowait(data)
            
            # Text response
            if text := response.text:
                eel.add_message("gemini", text)
            
            # User speech transcribed
            if response.server_content.input_transcription:
                transcript = response.server_content.input_transcription.text
                eel.add_user_transcription(transcript)
            
            # Gemini speech transcribed
            if response.server_content.output_transcription:
                transcript = response.server_content.output_transcription.text
                self.current_gemini_transcript.append(transcript)
                eel.add_gemini_transcription(transcript)
        
        # Turn complete - finalize message with frames
        if self.current_gemini_transcript:
            full_text = "".join(self.current_gemini_transcript)
            frame_data = self._prepare_frame_data()  # First & last frame
            eel.finalize_gemini_message(full_text, frame_data)
        
        # Clear audio queue (handles interruption)
        while not self.audio_in_queue.empty():
            self.audio_in_queue.get_nowait()
```

**Key insights**:
- Each turn is an async iterator of response chunks
- Transcripts arrive incrementally - concatenate them
- After turn ends, clear audio queue to handle interruptions
- Frame data is attached to completed messages

### 5. Video Narration (Lines ~290-330)

**What it does**: Periodically sends prompts to get AI commentary on video.

```python
async def video_narrator(self) -> None:
    prompt_index = 0
    is_first_narration = True
    
    while not self.should_stop:
        await asyncio.sleep(self.narration_interval)
        
        # Skip if no new frames since last narration
        if self.frames_sent_since_narration == 0:
            continue
        
        # First time: full description
        if is_first_narration and self.initial_prompt:
            prompt = self.initial_prompt
            is_first_narration = False
        else:
            # Subsequent: ask for changes (delta)
            prompt = self.narration_prompts[prompt_index % len(self.narration_prompts)]
            prompt_index += 1
        
        await self.session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=prompt)]),
            turn_complete=True  # Trigger immediate response
        )
        
        self.frames_sent_since_narration = 0  # Reset counter
```

**Key insights**:
- Uses `send_client_content()` for structured prompts
- First prompt gets baseline, subsequent prompts get deltas
- Only sends if new frames were captured
- `turn_complete=True` immediately triggers response

### 6. Audio Playback (Lines ~490-510)

**What it does**: Plays received audio through speakers.

```python
async def play_audio(self) -> None:
    stream = await asyncio.to_thread(
        pya.open,
        format=FORMAT,
        channels=CHANNELS,
        rate=RECEIVE_SAMPLE_RATE,  # 24kHz (API output rate)
        output=True
    )
    
    while not self.should_stop:
        bytestream = await self.audio_in_queue.get()
        await asyncio.to_thread(stream.write, bytestream)
```

**Key insight**: Audio playback is blocking, so use `asyncio.to_thread()`. The API always outputs at 24kHz regardless of input rate.

---

## Common Pitfalls and Solutions

### Pitfall 1: Blocking the Event Loop

**Problem**: Using synchronous operations directly in async functions:

```python
# WRONG - blocks event loop
async def capture_camera(self):
    cap = cv2.VideoCapture(0)  # Blocks!
    ret, frame = cap.read()    # Blocks!
```

**Solution**: Wrap in `asyncio.to_thread()`:

```python
# RIGHT - runs in thread pool
async def capture_camera(self):
    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
    ret, frame = await asyncio.to_thread(cap.read)
```

### Pitfall 2: Echo/Feedback Loop

**Problem**: Microphone picks up speaker output, model hears itself.

**Solution**: 
- Always use headphones
- Can't rely on software echo cancellation

### Pitfall 3: Incorrect Audio Format

**Problem**: Sending audio at wrong sample rate or format.

```python
# API requires 16kHz PCM for input
SEND_SAMPLE_RATE = 16000  # Must be 16kHz

# API outputs 24kHz PCM
RECEIVE_SAMPLE_RATE = 24000  # Always 24kHz
```

**Solution**: Use correct rates and 16-bit PCM format.

### Pitfall 4: Base64 Encoding Confusion

**Problem**: Video frames need special handling:

```python
# Capture returns base64-encoded string
frame = {"data": base64_string, "mime_type": "image/jpeg"}

# But API needs raw bytes
image_data = base64.b64decode(frame["data"])  # Decode first!
await session.send_realtime_input(
    media=types.Blob(data=image_data, mime_type=frame["mime_type"])
)
```

**Solution**: Always decode base64 back to bytes before creating Blob.

### Pitfall 5: Not Handling Interruption

**Problem**: Audio queue fills with old audio after user interrupts.

**Solution**: Clear queue when turn completes:

```python
# After turn ends
while not self.audio_in_queue.empty():
    self.audio_in_queue.get_nowait()
```

---

## Best Practices

### 1. Frame Rate Selection

```python
# Low bandwidth / cost-sensitive: 0.5 FPS
fps = 0.5

# Balanced (default): 1 FPS
fps = 1.0

# High responsiveness: 2-3 FPS
fps = 2.5

# Maximum (usually unnecessary): 5 FPS
fps = 5.0
```

Higher FPS = more responsive but uses more bandwidth and API quota.

### 2. System Instructions

Be specific and detailed:

```python
system_instruction = """
You are a yoga instructor. Your role is to:
- Observe the person's form and alignment
- Provide constructive feedback
- Emphasize safety over perfection
- Use encouraging, supportive language
"""
```

### 3. Narration Prompts

Use delta-based prompts (ask for changes, not full descriptions):

```python
narration_prompts = [
    "What changed since your last observation?",
    "Describe any differences in form or positioning.",
    "What improvements or issues do you notice?"
]
```

This reduces redundancy and keeps responses focused.

### 4. Queue Management

Set appropriate queue sizes:

```python
# Small queue for audio (low latency)
self.audio_in_queue = asyncio.Queue()

# Bounded queue for video (prevent memory bloat)
self.out_queue = asyncio.Queue(maxsize=20)
```

### 5. Error Handling

Always handle cleanup:

```python
try:
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        # Run tasks
        pass
except asyncio.CancelledError:
    pass  # Normal exit
except Exception as e:
    debug_log(f"Error: {e}", "ERROR")
finally:
    if self.audio_stream:
        self.audio_stream.close()
```

---

## Advanced Topics

### Turn Coverage

Controls which input data the model considers:

```python
# Include all input from session start
turn_coverage=TurnCoverage.TURN_INCLUDES_ALL_INPUT

# Only include input since last turn
turn_coverage=TurnCoverage.TURN_INCLUDES_INPUT_SINCE_LAST_TURN
```

The first option provides more context but may increase latency.

### Response Modalities

Control how the model responds:

```python
# Voice only
response_modalities=[Modality.AUDIO]

# Text only
response_modalities=[Modality.TEXT]

# Both
response_modalities=[Modality.AUDIO, Modality.TEXT]
```

### Custom VAD Thresholds

Not directly exposed in this API version, but VAD sensitivity can indirectly be influenced by:
- Audio gain/volume
- Background noise levels
- Speaking pace and pauses

---

## Summary

The Gemini Live API enables low-latency, bidirectional multimodal communication through:

1. **Persistent WebSocket connection** maintaining conversation state
2. **Concurrent task architecture** for non-blocking I/O
3. **Two input methods**: streaming (realtime_input) and structured (client_content)
4. **Automatic transcription** of all audio
5. **Voice Activity Detection** for natural turn-taking
6. **Interruption handling** for natural conversations

The key to building with Live API is understanding async/await patterns, managing concurrent tasks, and properly handling audio/video formats. This application demonstrates all these concepts in a production-ready architecture.