"""
Gemini Live API - Web UI Application (Standalone)

This version captures and displays:
1. Your voice input (transcribed automatically)
2. Gemini's voice output (transcribed automatically)
3. Text messages you type

NEW FEATURES:
- Configurable frame rate (0.5 - 5 FPS)
- Video narration mode for continuous AI feedback
- Extensive debug logging to diagnose video processing
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from textwrap import dedent

import cv2
import pyaudio
import eel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import LiveConnectConfig, Modality, TurnCoverage

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


# ============================================================================
# DEBUG LOGGING
# ============================================================================

DEBUG = True  # Set to False to disable debug output

def debug_log(message: str, category: str = "INFO") -> None:
    """Print debug messages with timestamp and category."""
    if DEBUG:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{category}] {message}")


# ============================================================================
# EEL INITIALIZATION
# ============================================================================
eel.init('web')


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODE = "camera"
DEFAULT_FPS = 1.0
DEFAULT_NARRATION_INTERVAL = 5.0  # seconds between narration prompts
MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"

# Predefined system prompts with matching narration prompts
SYSTEM_PROMPTS = {
    "video_describer": {
        "name": "Video Describer for Accessibility",
        "system_instruction": dedent("""\
            You are a compassionate and detailed video describer for people who are blind or have low vision. 
            Your role is to provide clear, comprehensive descriptions of everything visible in the video stream.

            Describe:
            - People: their appearance, clothing, facial expressions, and actions
            - Objects: what they are, their colors, sizes, and positions
            - Environment: the setting, lighting, and spatial layout
            - Movement: any motion or changes happening in the scene
            - Text: read any visible text, signs, or written content

            Be objective and descriptive. Prioritize the most relevant information first. Use clear, spatial 
            language (left, right, center, foreground, background). Avoid assumptions or interpretations - 
            describe only what you can see.
            """),
        "initial_prompt": "Describe everything important in the current scene - people, objects, environment, and activities.",
        "narration_prompts": [
            "What has changed since the last description? Describe any new elements, movements, or differences.",
            "Compare the current view to what you saw before. What's different?",
            "Describe any changes in the scene - what moved, appeared, or disappeared?"
        ]
    },

    "expression_analyzer": {
        "name": "Facial Expression Analyzer",
        "system_instruction": dedent("""\
            You are an expert facial expression and emotion analyzer. Your role is to observe faces in the 
            video and provide detailed analysis of expressions and apparent emotional states.

            Analyze and report:
            - Facial features: position of eyebrows, eyes, mouth, and overall muscle tension
            - Micro-expressions: subtle, brief emotional displays
            - Emotional indicators: what emotions the expressions suggest (happiness, sadness, surprise, 
              anger, fear, disgust, contempt, neutral)
            - Intensity: how strong the expression appears
            - Changes: how expressions shift over time

            Be specific and descriptive. Note that you're observing external expressions, which may not always 
            reflect internal feelings. Use tentative language like "appears to show" or "suggests" rather than 
            stating emotions as definite facts.
            """),
        "initial_prompt": "Analyze the facial expression you see right now. Describe the features and apparent emotional state in detail.",
        "narration_prompts": [
            "How has the facial expression changed since your last observation?",
            "Describe any shifts or transitions in emotional expression you notice.",
            "What changes do you see in the face compared to moments ago?"
        ]
    },

    "yoga_instructor": {
        "name": "Supportive Yoga Instructor",
        "system_instruction": dedent("""\
            You are a patient and encouraging yoga instructor focused on proper form, safety, and mindful 
            practice. Your role is to guide the person through yoga poses and provide constructive, supportive 
            feedback.

            Your approach:
            - Guide them into specific poses one at a time, clearly describing the proper form
            - Observe their alignment and positioning carefully
            - Provide gentle, constructive feedback on form and alignment
            - Emphasize safety and listening to their body above all else
            - Encourage them to modify poses as needed for their comfort and ability level
            - Use positive reinforcement while noting areas for improvement
            - Remind them to breathe and move mindfully

            Important: Always prioritize safety. If you notice potentially unsafe positioning, immediately 
            provide corrections. Never push someone beyond their apparent comfort level. Encourage rest when 
            needed.
            """),
        "initial_prompt": "Describe the person's current position and posture. What pose are they attempting?",
        "narration_prompts": [
            "What adjustments or changes has the person made to their pose since you last observed?",
            "How has their form or alignment changed? Note any improvements or areas needing attention.",
            "Describe any movement or position changes. Are they progressing in the pose or struggling?"
        ]
    }
}

# Live API configuration generator
def create_live_config(system_instruction: Optional[str] = None) -> LiveConnectConfig:
    """Create Live API configuration with optional system instruction."""
    config_dict = {
        "response_modalities": [Modality.AUDIO],
        "input_audio_transcription": types.AudioTranscriptionConfig(),
        "output_audio_transcription": types.AudioTranscriptionConfig(),
        "realtime_input_config": types.RealtimeInputConfig(turn_coverage=TurnCoverage.TURN_INCLUDES_ALL_INPUT)
    }

    if system_instruction:
        config_dict["system_instruction"] = system_instruction

    return types.LiveConnectConfig(**config_dict)

# PyAudio instance
pya = pyaudio.PyAudio()


# ============================================================================
# GLOBAL STATE
# ============================================================================

audio_loop: Optional['UIAudioLoop'] = None
is_running: bool = False
stop_requested: bool = False


# ============================================================================
# STANDALONE UI AUDIO LOOP (NO INHERITANCE)
# ============================================================================

class UIAudioLoop:
    """
    Standalone audio loop for web UI with configurable frame rate and narration mode.
    """

    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        fps: float = DEFAULT_FPS,
        narration_mode: bool = False,
        narration_interval: float = DEFAULT_NARRATION_INTERVAL,
        narration_prompts: Optional[List[str]] = None,
        initial_prompt: Optional[str] = None
    ) -> None:
        """
        Initialize the UI audio loop.

        Args:
            video_mode: "camera", "screen", or "none"
            fps: Frames per second (0.5 to 5.0)
            narration_mode: Enable automatic video narration
            narration_interval: Seconds between narration prompts
            narration_prompts: Custom list of narration prompts to cycle through
            initial_prompt: First prompt to establish baseline (before delta prompts)
        """
        # Configuration
        self.video_mode: str = video_mode
        self.fps: float = max(0.5, min(5.0, fps))
        self.frame_interval: float = 1.0 / self.fps
        self.narration_mode: bool = narration_mode
        self.narration_interval: float = narration_interval
        self.should_stop: bool = False

        # Initial prompt for first narration (full description)
        self.initial_prompt: Optional[str] = initial_prompt

        # Custom narration prompts (for subsequent delta updates)
        self.narration_prompts: List[str] = narration_prompts or [
            "What changed since your last observation?",
            "Describe any differences or movements you notice.",
            "What's new or different in the scene?"
        ]

        # Communication queues
        self.audio_in_queue: Optional[asyncio.Queue] = None
        self.out_queue: Optional[asyncio.Queue] = None

        # Session
        self.session: Optional[asyncio.Task] = None
        self.audio_stream: Optional[pyaudio.Stream] = None

        # Transcription tracking
        self.current_user_transcript: List[str] = []
        self.current_gemini_transcript: List[str] = []

        # Frame tracking for UI display
        self.latest_frame: Optional[Dict[str, str]] = None
        self.frame_count: int = 0
        self.turn_start_time: Optional[float] = None

        # Narration tracking
        self.last_narration_time: float = 0

        # Debug counters
        self.total_frames_captured: int = 0
        self.total_frames_sent: int = 0
        self.total_audio_chunks_sent: int = 0

        debug_log(f"Initialized: mode={video_mode}, fps={fps}, narration={narration_mode}, interval={narration_interval}s", "INIT")

    # ========================================================================
    # VIDEO CAPTURE TASKS
    # ========================================================================

    async def capture_camera(self) -> None:
        """Capture frames from webcam at configured FPS."""
        debug_log("Starting camera capture task", "VIDEO")
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while not self.should_stop:
            frame = await asyncio.to_thread(capture_camera_frame, cap)

            if frame is None:
                debug_log("Failed to capture camera frame", "VIDEO-ERROR")
                break

            self.latest_frame = frame
            self.frame_count += 1
            self.total_frames_captured += 1

            debug_log(f"Captured frame #{self.total_frames_captured} (turn frames: {self.frame_count})", "VIDEO")

            # Use configurable frame interval
            await asyncio.sleep(self.frame_interval)

            # Try to queue (may block if queue is full)
            queue_size = self.out_queue.qsize()
            debug_log(f"Queueing frame (queue size: {queue_size})", "VIDEO")

            await self.out_queue.put(frame)

        cap.release()
        debug_log("Camera capture task ended", "VIDEO")

    async def capture_screen(self) -> None:
        """Capture screen at configured FPS."""
        debug_log("Starting screen capture task", "VIDEO")

        while not self.should_stop:
            frame = await asyncio.to_thread(capture_screen_frame)

            if frame is None:
                debug_log("Failed to capture screen frame", "VIDEO-ERROR")
                break

            self.latest_frame = frame
            self.frame_count += 1
            self.total_frames_captured += 1

            debug_log(f"Captured screen frame #{self.total_frames_captured}", "VIDEO")

            # Use configurable frame interval
            await asyncio.sleep(self.frame_interval)

            queue_size = self.out_queue.qsize()
            debug_log(f"Queueing screen frame (queue size: {queue_size})", "VIDEO")

            await self.out_queue.put(frame)

        debug_log("Screen capture task ended", "VIDEO")

    # ========================================================================
    # VIDEO NARRATION TASK
    # ========================================================================

    async def video_narrator(self) -> None:
        """
        Periodically send prompts to get AI narration of video content.

        First prompt gets full baseline description, subsequent prompts ask for deltas.
        """
        debug_log("Starting video narrator task", "NARRATOR")
        debug_log(f"Using {len(self.narration_prompts)} narration prompts", "NARRATOR")

        prompt_index = 0
        is_first_narration = True

        while not self.should_stop:
            await asyncio.sleep(self.narration_interval)

            if self.should_stop:
                break

            current_time = time.time()

            # Only send if enough time has passed since last narration
            if current_time - self.last_narration_time >= self.narration_interval:
                try:
                    # First time: use initial prompt for full description
                    if is_first_narration and self.initial_prompt:
                        prompt = self.initial_prompt
                        is_first_narration = False
                        debug_log("Sending INITIAL narration prompt (baseline)", "NARRATOR")
                    else:
                        # Subsequent times: use delta prompts
                        prompt = self.narration_prompts[prompt_index % len(self.narration_prompts)]
                        prompt_index += 1
                        debug_log("Sending DELTA narration prompt", "NARRATOR")

                    debug_log(f"Prompt: '{prompt}'", "NARRATOR")
                    debug_log(f"Frames sent since last narration: {self.total_frames_sent}", "NARRATOR")

                    await self.session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        ),
                        turn_complete=True
                    )

                    self.last_narration_time = current_time

                    print(f"[Narration prompt sent: {prompt}]")

                except Exception as e:
                    debug_log(f"Failed to send narration prompt: {e}", "NARRATOR-ERROR")

        debug_log("Video narrator task ended", "NARRATOR")

    # ========================================================================
    # AUDIO INPUT TASK
    # ========================================================================

    async def listen_audio(self) -> None:
        """Capture audio from microphone and queue for sending."""
        debug_log("Starting audio capture task", "AUDIO")

        mic_info = pya.get_default_input_device_info()
        debug_log(f"Using microphone: {mic_info['name']}", "AUDIO")

        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while not self.should_stop:
            data = await asyncio.to_thread(
                self.audio_stream.read,
                CHUNK_SIZE,
                **kwargs
            )

            self.total_audio_chunks_sent += 1

            # Only log every 50 chunks to avoid spam
            if self.total_audio_chunks_sent % 50 == 0:
                debug_log(f"Captured {self.total_audio_chunks_sent} audio chunks", "AUDIO")

            await self.out_queue.put({
                "data": data,
                "mime_type": "audio/pcm"
            })

        debug_log("Audio capture task ended", "AUDIO")

    # ========================================================================
    # DATA SENDER TASK
    # ========================================================================

    async def send_realtime(self) -> None:
        """Send queued data to Gemini in real-time."""
        debug_log("Starting realtime sender task", "SENDER")

        while not self.should_stop:
            msg = await self.out_queue.get()

            mime_type = msg.get("mime_type", "")

            if "audio" in mime_type:
                # Audio data - don't log every chunk
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=msg["data"],
                        mime_type=mime_type
                    )
                )
            else:
                # Video frame
                import base64
                image_data = base64.b64decode(msg["data"])

                self.total_frames_sent += 1
                debug_log(f"Sending video frame #{self.total_frames_sent} to API", "SENDER")

                await self.session.send_realtime_input(
                    media=types.Blob(
                        data=image_data,
                        mime_type=mime_type
                    )
                )

        debug_log("Realtime sender task ended", "SENDER")

    # ========================================================================
    # RESPONSE RECEIVER TASK
    # ========================================================================

    async def receive_audio(self) -> None:
        """Receive responses from Gemini and send to UI."""
        debug_log("Starting response receiver task", "RECEIVER")

        while not self.should_stop:
            try:
                debug_log("Waiting for next turn from Gemini...", "RECEIVER")
                turn = self.session.receive()

                self.current_gemini_transcript = []
                self.frame_count = 0
                self.turn_start_time = time.time()

                debug_log("Turn started - processing responses", "RECEIVER")

                async for response in turn:
                    if self.should_stop:
                        break

                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue

                    if text := response.text:
                        debug_log(f"Received text response: {text[:50]}...", "RECEIVER")
                        try:
                            eel.add_message("gemini", text)
                        except Exception as e:
                            print(f"Failed to update UI: {e}")
                        print(text, end="")
                        continue

                    if response.server_content.input_transcription:
                        transcript_text = response.server_content.input_transcription.text
                        self.current_user_transcript.append(transcript_text)

                        debug_log(f"User transcript: {transcript_text}", "RECEIVER")

                        try:
                            eel.add_user_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update user transcript: {e}")

                        print(f"\n[You: {transcript_text}]", end="", flush=True)

                    if response.server_content.output_transcription:
                        transcript_text = response.server_content.output_transcription.text
                        self.current_gemini_transcript.append(transcript_text)

                        try:
                            eel.add_gemini_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update Gemini transcript: {e}")

                        print(transcript_text, end="", flush=True)

                # TURN END
                debug_log("Turn ended - finalizing transcripts", "RECEIVER")

                if self.current_user_transcript:
                    full_user_text = "".join(self.current_user_transcript)
                    try:
                        eel.finalize_user_message(full_user_text)
                    excep