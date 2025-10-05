# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gemini Live API - Real-time Multimodal Interaction Script

This script demonstrates how to use the Gemini Live API for real-time bidirectional
communication with Google's Gemini AI model. The Live API enables low-latency
interactions with audio, video, and text streaming.

## Key Concepts:

**Session**: A persistent WebSocket connection that maintains conversation state
**Turn**: A single exchange where user provides input and model responds
**Voice Activity Detection (VAD)**: Automatic detection of when user starts/stops speaking
**Interruption**: User can interrupt the model mid-response by speaking
**Response Modalities**: Output format - AUDIO (spoken) or TEXT (written)

## Setup

To install the dependencies for this script, run:

```
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones.

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
Gemini Live API - Console Application

WHAT THIS APP DOES:
A command-line interface for real-time conversations with Gemini using:
- Voice input (your microphone)
- Voice output (your speakers)
- Optional video input (camera or screen)
- Text input (keyboard)
- Real-time transcription of all audio

USAGE:
    python main.py                    # Camera mode (default)
    python main.py --mode screen      # Screen share mode
    python main.py --mode none        # Audio only mode

REQUIREMENTS:
- GOOGLE_API_KEY environment variable
- Headphones (to prevent echo/feedback)
- Microphone and speakers
- Optional: Webcam for video mode
"""

import asyncio
import argparse
import sys
import traceback
from typing import Optional

import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, Modality

# Import shared utilities
from utils import (
    capture_camera_frame,
    capture_screen_frame,
    FORMAT,
    CHANNELS,
    SEND_SAMPLE_RATE,
    RECEIVE_SAMPLE_RATE,
    CHUNK_SIZE
)

# Load environment variables (.env file)
load_dotenv()

# Python 3.11+ has TaskGroup built-in
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# ============================================================================
# CONFIGURATION
# ============================================================================

# Gemini model optimized for Live API
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

# Default video mode
DEFAULT_MODE = "camera"

# Live API configuration
CONFIG = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig()
)

# Initialize Gemini client
client = genai.Client(http_options={"api_version": "v1beta"})

# Initialize PyAudio
pya = pyaudio.PyAudio()


# ============================================================================
# MAIN AUDIO LOOP CLASS
# ============================================================================

class AudioLoop:
    """
    Manages real-time bidirectional communication with Gemini Live API.

    ARCHITECTURE:
    Runs 6 concurrent tasks:
    1. send_text() - Accept keyboard input
    2. send_realtime() - Send queued data to Gemini
    3. listen_audio() - Capture microphone
    4. get_frames() or get_screen() - Capture video (optional)
    5. receive_audio() - Receive Gemini responses
    6. play_audio() - Play audio through speakers

    COMMUNICATION:
    Tasks communicate via asyncio Queues:
    - out_queue: Data going TO Gemini (audio, video, images)
    - audio_in_queue: Audio coming FROM Gemini (to be played)
    """

    def __init__(self, video_mode: str = DEFAULT_MODE) -> None:
        """
        Initialize the audio loop.

        Args:
            video_mode: "camera", "screen", or "none"
        """
        self.video_mode: str = video_mode

        # Queues for inter-task communication
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        # Live API session (WebSocket connection)
        self.session: Optional[asyncio.Task] = None

        # Audio stream (for cleanup)
        self.audio_stream: Optional[pyaudio.Stream] = None

    # ========================================================================
    # TEXT INPUT TASK
    # ========================================================================

    async def send_text(self) -> None:
        """
        Accept keyboard input and send to Gemini.

        STRUCTURED TURNS:
        Uses send_client_content() which creates explicit turn boundaries.
        This is different from send_realtime_input() which streams continuously.

        TURN COMPLETION:
        Setting turn_complete=True signals to Gemini:
        "I'm done with my input, please respond now"

        EXIT:
        Type 'q' to quit the conversation.
        """
        while True:
            try:
                # Read input (blocking, so use thread)
                text = await asyncio.to_thread(input, "message > ")

                if text.lower() == "q":
                    # User wants to exit
                    break

                # Send to Gemini as a complete turn
                await self.session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text=text or ".")]
                    ),
                    turn_complete=True  # Triggers response
                )

            except (KeyboardInterrupt, EOFError, UnicodeDecodeError):
                print("\nExiting...")
                break

    # ========================================================================
    # VIDEO CAPTURE TASKS
    # ========================================================================

    async def get_frames(self) -> None:
        """
        Continuously capture and send camera frames.

        FRAME RATE:
        Captures at 1 FPS (one frame per second).
        This matches Gemini's processing rate.

        THREADING:
        Frame capture is blocking, so we use asyncio.to_thread()
        to prevent blocking the audio pipeline.
        """
        import cv2

        # Open camera (blocking operation)
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while True:
            # Capture frame (blocking operation)
            frame = await asyncio.to_thread(capture_camera_frame, cap)

            if frame is None:
                # Camera failed
                break

            # Wait 1 second between frames
            await asyncio.sleep(1.0)

            # Queue frame to be sent to Gemini
            await self.out_queue.put(frame)

        # Cleanup
        cap.release()

    async def get_screen(self) -> None:
        """
        Continuously capture and send screen frames.

        Same as get_frames() but captures screen instead of camera.
        """
        while True:
            # Capture screen (blocking operation)
            frame = await asyncio.to_thread(capture_screen_frame)

            if frame is None:
                # Screen capture failed
                break

            # Wait 1 second between frames
            await asyncio.sleep(1.0)

            # Queue frame to be sent to Gemini
            await self.out_queue.put(frame)

    # ========================================================================
    # REALTIME DATA SENDER TASK
    # ========================================================================

    async def send_realtime(self) -> None:
        """
        Send queued data (audio/video) to Gemini in real-time.

        STREAMING VS STRUCTURED:
        This uses send_realtime_input() which:
        - Streams data continuously without explicit turns
        - Optimized for low latency
        - Relies on VAD (Voice Activity Detection) for turn boundaries
        - Can be sent while model is generating

        AUDIO VS MEDIA:
        - Audio: Sent via audio parameter (raw PCM bytes)
        - Images/Video: Sent via media parameter (base64 → bytes)
        """
        while True:
            # Wait for data in queue
            msg = await self.out_queue.get()

            mime_type = msg.get("mime_type", "")

            if "audio" in mime_type:
                # Audio data (raw PCM from microphone)
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=msg["data"],
                        mime_type=mime_type
                    )
                )
            else:
                # Image/video data (base64 encoded)
                # Decode base64 back to bytes for Blob
                import base64
                image_data = base64.b64decode(msg["data"])

                await self.session.send_realtime_input(
                    media=types.Blob(
                        data=image_data,
                        mime_type=mime_type
                    )
                )

    # ========================================================================
    # AUDIO INPUT TASK
    # ========================================================================

    async def listen_audio(self) -> None:
        """
        Capture audio from microphone and queue for sending.

        AUDIO FORMAT:
        - 16-bit PCM (raw uncompressed audio)
        - 16kHz sample rate (required by Live API)
        - Mono (single channel)
        - 1024 frames per buffer (~64ms latency)

        VOICE ACTIVITY DETECTION:
        Gemini automatically detects when you:
        - Start speaking (turn begins)
        - Stop speaking (turn ends, triggers response)
        """
        # Get default microphone
        mic_info = pya.get_default_input_device_info()

        # Open audio stream
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        # Prevent exceptions on buffer overflow (can happen if system is slow)
        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while True:
            # Read audio chunk (blocking, so use thread)
            data = await asyncio.to_thread(
                self.audio_stream.read,
                CHUNK_SIZE,
                **kwargs
            )

            # Queue for sending to Gemini
            await self.out_queue.put({
                "data": data,
                "mime_type": "audio/pcm"
            })

    # ========================================================================
    # RESPONSE RECEIVER TASK
    # ========================================================================

    async def receive_audio(self) -> None:
        """
        Receive and process responses from Gemini.

        TURN STRUCTURE:
        Each turn contains multiple response chunks:
        - response.data: Audio bytes (PCM at 24kHz)
        - response.text: Text content (if TEXT mode)
        - response.server_content.input_transcription: Your speech as text
        - response.server_content.output_transcription: Gemini's speech as text

        INTERRUPTION:
        When you interrupt (start speaking while Gemini is talking):
        - Gemini sends turn_complete
        - We clear the audio queue (stop playing old audio)
        - Your new input takes priority
        """
        while True:
            # Get next turn
            turn = self.session.receive()

            # Process all response chunks in this turn
            async for response in turn:
                # Audio to play
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue

                # Text response (TEXT mode only)
                if text := response.text:
                    print(text, end="")
                    continue

                # Input transcription (what you said)
                if response.server_content.input_transcription:
                    transcript = response.server_content.input_transcription.text
                    print(f"\n[You: {transcript}]", end="", flush=True)

                # Output transcription (what Gemini is saying)
                if response.server_content.output_transcription:
                    transcript = response.server_content.output_transcription.text
                    print(transcript, end="", flush=True)

            # Turn complete - clear audio queue (handles interruption)
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    # ========================================================================
    # AUDIO OUTPUT TASK
    # ========================================================================

    async def play_audio(self) -> None:
        """
        Play audio responses through speakers.

        AUDIO FORMAT:
        - 16-bit PCM (same as input)
        - 24kHz sample rate (Gemini always outputs at this rate)
        - Mono (single channel)

        PLAYBACK:
        Audio chunks are queued and played as fast as they arrive.
        The queue ensures smooth playback even with network jitter.
        """
        # Open audio stream for playback
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while True:
            # Wait for audio from Gemini
            bytestream = await self.audio_in_queue.get()

            # Play through speakers (blocking, so use thread)
            await asyncio.to_thread(stream.write, bytestream)

    # ========================================================================
    # MAIN RUN METHOD
    # ========================================================================

    async def run(self) -> None:
        """
        Main entry point - sets up and runs all tasks.

        TASK GROUP:
        All tasks run concurrently in a TaskGroup:
        - If any task fails, all are cancelled
        - Clean shutdown when user types 'q'

        SESSION:
        The Live API session is a WebSocket connection that:
        - Maintains conversation context
        - Handles VAD automatically
        - Supports interruption
        - Has a 10-minute default timeout (configurable)
        """
        try:
            # Create session and task group
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Initialize queues
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Create all concurrent tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Start video capture (if enabled)
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Wait for user to exit (type 'q')
                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            # Normal exit
            pass
        except ExceptionGroup as EG:
            # Error in one of the tasks
            if self.audio_stream:
                self.audio_stream.close()
            traceback.print_exception(EG)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Gemini Live API Console Application"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["camera", "screen", "none"],
        help="Video mode: camera, screen, or none (audio only)"
    )
    args = parser.parse_args()

    # Create and run the audio loop
    loop = AudioLoop(video_mode=args.mode)
    asyncio.run(loop.run())