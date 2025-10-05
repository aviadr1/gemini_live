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
"""

import asyncio
import base64
import io
import os
import sys
import traceback
from typing import Optional, Dict, Any

import cv2
import pyaudio
import PIL.Image
import mss

import argparse

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file (including GOOGLE_API_KEY)
load_dotenv()

# Python 3.11+ has TaskGroup built-in, but older versions need backports
if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio format constants for PyAudio
# These match the required format for Gemini Live API
FORMAT = pyaudio.paInt16  # 16-bit PCM audio format
CHANNELS = 1  # Mono audio (single channel)
SEND_SAMPLE_RATE = 16000  # Input audio: 16kHz (required by Live API)
RECEIVE_SAMPLE_RATE = 24000  # Output audio: 24kHz (Live API always outputs at this rate)
CHUNK_SIZE = 1024  # Number of audio frames per buffer (affects latency)

# Gemini model to use - this is the Live API optimized model
# "flash-live" models support real-time bidirectional streaming
MODEL = "models/gemini-2.0-flash-live-001"

# Default video mode - can be "camera", "screen", or "none"
DEFAULT_MODE = "camera"

# Initialize the Gemini client with API version
# The client handles authentication using GOOGLE_API_KEY environment variable
client = genai.Client(http_options={"api_version": "v1beta"})

# Configuration for the Live API session
# response_modalities determines the format of model's responses
# ["AUDIO"] means the model will respond with spoken audio
# Could also be ["TEXT"] for text-only responses
CONFIG = {"response_modalities": ["AUDIO"]}

# Initialize PyAudio for handling audio input/output
pya = pyaudio.PyAudio()


class AudioLoop:
    """
    Main class that manages the bidirectional audio/video streaming with Gemini Live API.

    This class orchestrates multiple concurrent tasks:
    1. Listening to microphone input (user speaking)
    2. Capturing video frames (from camera or screen)
    3. Sending audio/video to Gemini API
    4. Receiving audio responses from Gemini API
    5. Playing audio responses through speakers
    6. Accepting text input from console

    The Live API uses a "turn-based" conversation model where:
    - User provides input (audio, video, or text)
    - Model processes and generates a response
    - User can interrupt the model at any time by speaking
    """

    def __init__(self, video_mode: str = DEFAULT_MODE) -> None:
        """
        Initialize the AudioLoop with specified video mode.

        Args:
            video_mode: One of "camera", "screen", or "none"
                       - "camera": Captures video from webcam
                       - "screen": Captures screen content
                       - "none": Audio-only mode
        """
        self.video_mode = video_mode

        # Queues for managing data flow between tasks
        # These enable asynchronous communication between different parts of the system
        self.audio_in_queue = None  # Queue for audio coming FROM Gemini (to be played)
        self.out_queue = None  # Queue for data going TO Gemini (audio/video/text)

        # The Live API session object - represents the WebSocket connection
        self.session = None

        # Task references (not actively used but kept for potential cleanup)
        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None

    async def send_text(self) -> None:
        """
        Console input task - allows user to type messages to Gemini.

        This runs in a separate task and continuously waits for user input.
        When user types a message and presses Enter, it sends that message
        to Gemini and marks it as end_of_turn=True.

        **Turn Management**:
        - end_of_turn=True signals that the user has completed their input
        - This triggers Gemini to start processing and generating a response
        - Without this flag, Gemini waits for more input

        Type 'q' to quit the conversation.
        """
        while True:
            # Use asyncio.to_thread to prevent blocking the event loop
            # input() is a blocking call, so we run it in a separate thread
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break

            # Send text to Gemini Live API
            # - input: The text message to send
            # - end_of_turn: Signals that user has finished their turn
            #   This tells Gemini: "I'm done talking, your turn to respond"
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap: cv2.VideoCapture) -> Optional[Dict[str, str]]:
        """
        Capture a single frame from the camera and prepare it for Gemini.

        Args:
            cap: OpenCV VideoCapture object for the camera

        Returns:
            Dictionary with MIME type and base64-encoded JPEG data,
            or None if frame capture failed

        **Frame Processing**:
        1. Capture raw frame from camera
        2. Convert BGR (OpenCV default) to RGB (expected by Gemini)
        3. Resize to max 1024x1024 (to reduce bandwidth)
        4. Encode as JPEG
        5. Base64 encode for transmission
        """
        # Read the frame from camera
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self) -> None:
        """
        Continuously capture frames from camera and send to Gemini.

        **Video Streaming**:
        - Captures at ~1 FPS (one frame per second)
        - Live API processes video at 1 FPS by default
        - Each frame is added to out_queue to be sent to Gemini
        - Gemini "sees" the video and can reference it in responses

        **Important**: Frame capture blocks for ~1 second, so we use
        asyncio.to_thread to prevent blocking the audio pipeline.
        """
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            # Wait 1 second between frames (1 FPS)
            await asyncio.sleep(1.0)

            # Add frame to queue to be sent to Gemini
            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self) -> Dict[str, str]:
        """
        Capture the entire screen and prepare it for Gemini.

        Returns:
            Dictionary with MIME type and base64-encoded JPEG data

        Similar to _get_frame but captures screen instead of camera.
        Useful for screen sharing, presentations, or showing Gemini
        what you're looking at on your computer.
        """
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self) -> None:
        """
        Continuously capture screen and send to Gemini.

        Same concept as get_frames() but for screen capture.
        Allows Gemini to "see" your screen in real-time.
        """

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self) -> None:
        """
        Send queued data (audio/video/text) to Gemini in real-time.

        This task continuously pulls data from out_queue and sends it
        to the Gemini Live API session.

        **Real-time Streaming**:
        - Audio chunks are sent as soon as they're captured (~every 64ms)
        - Video frames are sent every 1 second
        - Text messages are sent when user presses Enter

        **No Turn Boundaries**: This method does NOT set end_of_turn,
        so Gemini receives continuous streaming data. The model uses
        Voice Activity Detection (VAD) to determine when user stops speaking.
        """
        while True:
            # Wait for data to appear in queue
            msg = await self.out_queue.get()
            # Send immediately to Gemini
            await self.session.send(input=msg)

    async def listen_audio(self) -> None:
        """
        Capture audio from microphone and queue it for sending to Gemini.

        This task:
        1. Opens the system's default microphone
        2. Continuously reads audio chunks (1024 frames at a time)
        3. Puts raw PCM audio data into out_queue
        4. Data is then sent to Gemini by send_realtime()

        **Audio Format**:
        - 16-bit PCM (Pulse Code Modulation) - raw uncompressed audio
        - 16kHz sample rate (required by Live API for input)
        - Mono (single channel)

        **Voice Activity Detection**:
        - Gemini automatically detects when you start/stop speaking
        - When you stop speaking, Gemini knows your turn is complete
        - It then starts generating a response
        """
        # Get system's default microphone info
        mic_info = pya.get_default_input_device_info()
        # Open audio stream for recording
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        # In debug mode, don't raise exception on buffer overflow
        # (can happen if system is slow)
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            # Read audio chunk from microphone (blocking call, so use to_thread)
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            # Queue the audio data to be sent to Gemini
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self) -> None:
        """
        Receive responses from Gemini and manage turn completion.

        This is the CORE task for understanding Gemini's response cycle.

        **How Turns Work**:
        1. User speaks (audio sent via listen_audio → send_realtime)
        2. User stops speaking (detected by VAD)
        3. Gemini starts generating response
        4. Response arrives as stream of audio chunks
        5. turn_complete signal indicates Gemini finished its response

        **Response Types**:
        - response.data: Audio chunk (PCM bytes) to be played
        - response.text: Transcript of what Gemini is saying
        - response.server_content.interrupted: User interrupted the model
        - response.server_content.turn_complete: Model finished its turn

        **Interruption Handling**:
        When user interrupts (starts speaking while model is talking):
        1. Gemini sends interrupted=True
        2. We clear the audio queue (stop playing the rest)
        3. This allows immediate response to user's new input

        This task reads from the websocket and writes PCM chunks to the output queue.
        """
        while True:
            # Get the next "turn" from Gemini
            # A turn is one complete response from the model
            turn = self.session.receive()

            # Iterate through all response chunks in this turn
            async for response in turn:
                # If there's audio data, queue it for playback
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                # If there's text (transcript), print it
                if text := response.text:
                    print(text, end="")

            # **CRITICAL INTERRUPTION HANDLING**:
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            #
            # Without this, if you interrupt Gemini mid-sentence,
            # it would keep playing the rest of that sentence even
            # though you've already started talking about something else.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self) -> None:
        """
        Play audio responses from Gemini through speakers.

        This task:
        1. Waits for audio chunks in audio_in_queue
        2. Plays them through the system's default speakers

        **Audio Playback**:
        - 24kHz sample rate (Gemini always outputs at this rate)
        - 16-bit PCM format
        - Mono audio

        This runs continuously, playing audio as fast as it arrives.
        The queue ensures smooth playback even if network is choppy.
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
            # Wait for audio data from Gemini
            bytestream = await self.audio_in_queue.get()
            # Play it through speakers (blocking call, so use to_thread)
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self) -> None:
        """
        Main entry point that orchestrates all tasks.

        **Task Architecture**:
        This creates 6 concurrent tasks that run simultaneously:

        INPUT TASKS (user → Gemini):
        1. send_text(): Accept typed messages from console
        2. listen_audio(): Capture microphone audio
        3. get_frames()/get_screen(): Capture video
        4. send_realtime(): Send all queued data to Gemini

        OUTPUT TASKS (Gemini → user):
        5. receive_audio(): Receive responses from Gemini
        6. play_audio(): Play audio through speakers

        **Session Management**:
        - client.aio.live.connect() creates a WebSocket connection
        - This connection persists for the entire conversation
        - Default session limit is 10 minutes (can be extended)

        **TaskGroup**:
        - All tasks run concurrently
        - If one task fails, all others are cancelled
        - Clean shutdown when user types 'q'
        """
        try:
            # Create Live API session (WebSocket connection to Gemini)
            # and TaskGroup for managing concurrent tasks
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                # Initialize queues for inter-task communication
                self.audio_in_queue = asyncio.Queue()  # Gemini audio → speakers
                self.out_queue = asyncio.Queue(maxsize=5)  # User input → Gemini

                # Create all concurrent tasks
                # send_text is the main task - when it completes (user types 'q'),
                # we exit and cancel all other tasks
                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Conditionally start video capture based on mode
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
            # Normal exit - user typed 'q'
            pass
        except ExceptionGroup as EG:
            # Handle any errors from the tasks
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    # Create and run the audio loop
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())