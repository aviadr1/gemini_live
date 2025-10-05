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
import base64
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
        self.frames_sent_since_narration: int = 0

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
                # Skip if no frames were sent since last narration
                if self.frames_sent_since_narration == 0:
                    debug_log("Skipping narration - no new frames captured", "NARRATOR")
                    continue

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
                    debug_log(f"Frames sent since last narration: {self.frames_sent_since_narration}", "NARRATOR")

                    await self.session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        ),
                        turn_complete=True
                    )

                    self.last_narration_time = current_time
                    self.frames_sent_since_narration = 0  # Reset counter after sending narration

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
                image_data = base64.b64decode(msg["data"])

                self.total_frames_sent += 1
                self.frames_sent_since_narration += 1  # Track frames since last narration
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
                    except Exception as e:
                        print(f"Failed to finalize user message: {e}")
                    self.current_user_transcript = []

                if self.current_gemini_transcript:
                    full_gemini_text = "".join(self.current_gemini_transcript)

                    frame_data: Optional[Dict[str, Any]] = None

                    if self.video_mode != "none" and self.latest_frame:
                        turn_duration = time.time() - self.turn_start_time if self.turn_start_time else 0

                        frame_data = {
                            "image": self.latest_frame["data"],
                            "mime_type": self.latest_frame["mime_type"],
                            "frame_count": max(1, self.frame_count),
                            "duration": round(turn_duration, 1)
                        }

                        debug_log(f"Attaching {self.frame_count} frames to response", "RECEIVER")

                    try:
                        eel.finalize_gemini_message(full_gemini_text, frame_data)
                    except Exception as e:
                        print(f"Failed to finalize Gemini message: {e}")

                    self.current_gemini_transcript = []

                # Clear audio queue
                queue_size = self.audio_in_queue.qsize()
                if queue_size > 0:
                    debug_log(f"Clearing {queue_size} audio items from queue", "RECEIVER")

                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

            except Exception as e:
                if not self.should_stop:
                    debug_log(f"Error in receive_audio: {e}", "RECEIVER-ERROR")
                    traceback.print_exc()
                break

        debug_log("Response receiver task ended", "RECEIVER")

    # ========================================================================
    # AUDIO OUTPUT TASK
    # ========================================================================

    async def play_audio(self) -> None:
        """Play audio responses through speakers."""
        debug_log("Starting audio playback task", "PLAYBACK")

        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while not self.should_stop:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

        debug_log("Audio playback task ended", "PLAYBACK")

    # ========================================================================
    # WAIT TASK
    # ========================================================================

    async def wait_for_stop(self) -> None:
        """Wait for stop signal."""
        while not self.should_stop:
            await asyncio.sleep(0.1)


# ============================================================================
# EEL-EXPOSED FUNCTIONS
# ============================================================================

@eel.expose
def start_session(
    mode: str = DEFAULT_MODE,
    fps: float = DEFAULT_FPS,
    narration_mode: bool = False,
    narration_interval: float = DEFAULT_NARRATION_INTERVAL,
    system_prompt_key: str = "",
    custom_system_prompt: str = "",
    custom_narration_prompts: str = ""
) -> Dict[str, str]:
    """Start a new Gemini Live API session with configuration."""
    global audio_loop, is_running, stop_requested

    if is_running:
        return {"status": "error", "message": "Session already running"}

    try:
        # Determine system instruction and narration prompts
        system_instruction = None
        narration_prompts = None
        initial_prompt = None

        if system_prompt_key and system_prompt_key in SYSTEM_PROMPTS:
            # Use predefined prompt
            prompt_config = SYSTEM_PROMPTS[system_prompt_key]
            system_instruction = prompt_config["system_instruction"]
            initial_prompt = prompt_config["initial_prompt"]
            narration_prompts = prompt_config["narration_prompts"]
            debug_log(f"Using predefined prompt: {prompt_config['name']}", "SESSION")
        elif custom_system_prompt:
            # Use custom prompt
            system_instruction = custom_system_prompt
            debug_log("Using custom system prompt", "SESSION")

            # Parse custom narration prompts (one per line)
            if custom_narration_prompts:
                prompts_list = [
                    p.strip() for p in custom_narration_prompts.split('\n')
                    if p.strip()
                ]
                # First line is initial prompt, rest are delta prompts
                if prompts_list:
                    initial_prompt = prompts_list[0]
                    narration_prompts = prompts_list[1:] if len(prompts_list) > 1 else prompts_list
                    debug_log(f"Using custom initial prompt + {len(narration_prompts)} delta prompts", "SESSION")

        print(f"Starting session: mode={mode}, fps={fps}, narration={narration_mode}")
        stop_requested = False
        audio_loop = UIAudioLoop(
            video_mode=mode,
            fps=fps,
            narration_mode=narration_mode,
            narration_interval=narration_interval,
            narration_prompts=narration_prompts,
            initial_prompt=initial_prompt
        )

        import threading

        def run_async_session() -> None:
            global is_running

            time.sleep(0.5)

            try:
                debug_log("Connecting to Gemini Live API...", "SESSION")
                is_running = True

                status_msg = f"Connected - {mode.title()} @ {fps} FPS"
                if narration_mode:
                    status_msg += f" | Narration every {narration_interval}s"
                eel.update_status(status_msg)

                client = genai.Client(http_options={"api_version": "v1beta"})

                async def run_with_config() -> None:
                    debug_log("Opening Live API connection...", "SESSION")

                    # Create config with system instruction
                    config = create_live_config(system_instruction)

                    async with (
                        client.aio.live.connect(model=MODEL, config=config) as session,
                        asyncio.TaskGroup() as tg,
                    ):
                        audio_loop.session = session
                        audio_loop.audio_in_queue = asyncio.Queue()
                        audio_loop.out_queue = asyncio.Queue(maxsize=20)

                        debug_log("Session established, creating tasks...", "SESSION")

                        wait_task = tg.create_task(audio_loop.wait_for_stop())
                        tg.create_task(audio_loop.send_realtime())
                        tg.create_task(audio_loop.listen_audio())

                        if audio_loop.video_mode == "camera":
                            tg.create_task(audio_loop.capture_camera())
                        elif audio_loop.video_mode == "screen":
                            tg.create_task(audio_loop.capture_screen())

                        # Start video narrator if enabled
                        if audio_loop.narration_mode:
                            tg.create_task(audio_loop.video_narrator())

                        tg.create_task(audio_loop.receive_audio())
                        tg.create_task(audio_loop.play_audio())

                        debug_log("All tasks started", "SESSION")

                        await wait_task

                asyncio.run(run_with_config())

            except Exception as e:
                debug_log(f"Session error: {e}", "SESSION-ERROR")
                traceback.print_exc()
                try:
                    eel.update_status(f"Error: {str(e)}")
                except:
                    pass
            finally:
                is_running = False
                debug_log("Session ended", "SESSION")
                try:
                    eel.update_status("Disconnected")
                except:
                    pass

        thread = threading.Thread(target=run_async_session, daemon=True)
        thread.start()

        return {"status": "success", "message": "Session started"}

    except Exception as e:
        debug_log(f"Failed to start session: {e}", "SESSION-ERROR")
        traceback.print_exc()
        is_running = False
        return {"status": "error", "message": str(e)}


@eel.expose
def stop_session() -> Dict[str, str]:
    """Stop the current session."""
    global audio_loop, is_running, stop_requested

    if not is_running:
        return {"status": "error", "message": "No session running"}

    debug_log("Stop requested by user", "SESSION")
    stop_requested = True

    try:
        eel.update_status("Stopping...")
    except:
        pass

    if audio_loop:
        audio_loop.should_stop = True

    return {"status": "success", "message": "Stopping session..."}


@eel.expose
def send_message(text: str) -> Dict[str, str]:
    """Send a text message to Gemini."""
    global audio_loop, is_running

    if not is_running or not audio_loop or not audio_loop.session:
        return {"status": "error", "message": "No active session"}

    try:
        debug_log(f"Sending text message: '{text}'", "MESSAGE")

        eel.add_message("user", text)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        loop.run_until_complete(
            audio_loop.session.send_client_content(
                turns=types.Content(
                    role="user",
                    parts=[types.Part(text=text)]
                ),
                turn_complete=True
            )
        )

        loop.close()
        return {"status": "success"}

    except Exception as e:
        debug_log(f"Failed to send message: {e}", "MESSAGE-ERROR")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@eel.expose
def get_status() -> Dict[str, Any]:
    """Get current session status."""
    return {
        "is_running": is_running,
        "mode": audio_loop.video_mode if audio_loop else None
    }


@eel.expose
def get_system_prompts() -> Dict[str, Any]:
    """Get available system prompts for UI dropdown."""
    prompts = {}
    for key, config in SYSTEM_PROMPTS.items():
        prompts[key] = {
            "name": config["name"],
            "system_instruction": config["system_instruction"],
            "initial_prompt": config["initial_prompt"],
            "narration_prompts": config["narration_prompts"]
        }
    return prompts


# ============================================================================
# WEB FILE SYNCHRONIZATION
# ============================================================================

def sync_web_files() -> None:
    """Sync web files from templates to web directory."""
    web_dir = Path(__file__).parent / 'web'
    web_dir.mkdir(exist_ok=True)

    templates_dir = Path(__file__).parent / 'web_templates'

    if not templates_dir.exists():
        print(f"Error: web_templates directory not found at {templates_dir}")
        sys.exit(1)

    updated_files: List[str] = []

    for filename in ['index.html', 'style.css', 'script.js']:
        src = templates_dir / filename
        dst = web_dir / filename

        if not src.exists():
            print(f"Error: {filename} not found in web_templates/")
            sys.exit(1)

        should_copy = False
        reason = ""

        if not dst.exists():
            should_copy = True
            reason = "new file"
        elif src.stat().st_mtime > dst.stat().st_mtime:
            should_copy = True
            reason = "template is newer"

        if should_copy:
            dst.write_text(src.read_text())
            updated_files.append(f"{filename} ({reason})")

    if updated_files:
        print(f"✓ Updated web files: {', '.join(updated_files)}")
    else:
        print("✓ Web files are up to date")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Main application entry point."""
    sync_web_files()

    try:
        eel.start('index.html', size=(1000, 800), mode='default')
    except (SystemExit, KeyboardInterrupt):
        print("\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    load_dotenv()
    main()