"""
Gemini Live API - Web UI Application (Standalone)

This version captures and displays:
1. Your voice input (transcribed automatically)
2. Gemini's voice output (transcribed automatically)
3. Text messages you type
This is the web UI version with NO inheritance - completely standalone.
Uses utility functions from utils.py for shared functionality.

KEY DIFFERENCE FROM CONSOLE VERSION:
- Sends transcripts to browser UI in real-time
- Tracks captured frames for visual display
- Manages session via Eel (Python ↔ JavaScript bridge)
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

import cv2
import pyaudio
import eel
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


# ============================================================================
# EEL INITIALIZATION
# ============================================================================
eel.init('web')


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODE = "camera"
MODEL = "models/gemini-2.0-flash-live-001"

# Live API configuration with transcription enabled
UI_CONFIG: LiveConnectConfig = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig()
)

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
    Standalone audio loop for web UI - no inheritance, completely self-contained.

    RESPONSIBILITIES:
    - Capture audio from microphone
    - Capture video/screen frames
    - Send data to Gemini Live API
    - Receive and play audio responses
    - Send transcripts to browser UI
    - Track frames for visual display

    DESIGN:
    Uses composition instead of inheritance - imports utility functions
    for frame capture rather than inheriting from a parent class.
    """

    def __init__(self, video_mode: str = DEFAULT_MODE) -> None:
        """
        Initialize the UI audio loop.

        Args:
            video_mode: "camera", "screen", or "none"
        """
        # Configuration
        self.video_mode: str = video_mode
        self.should_stop: bool = False

        # Communication queues
        self.audio_in_queue: Optional[asyncio.Queue] = None  # Gemini → speakers
        self.out_queue: Optional[asyncio.Queue] = None  # User → Gemini

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

    # ========================================================================
    # VIDEO CAPTURE TASKS
    # ========================================================================

    async def capture_camera(self) -> None:
        """
        Capture frames from webcam and track for UI display.

        DIFFERENCES FROM CONSOLE VERSION:
        - Tracks latest_frame for UI display
        - Counts frames per turn
        - Same frame capture logic (via utility function)
        """
        # Open camera
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while not self.should_stop:
            # Use utility function for frame capture
            frame = await asyncio.to_thread(capture_camera_frame, cap)

            if frame is None:
                break

            # Track for UI display
            self.latest_frame = frame
            self.frame_count += 1

            # Wait 1 second (1 FPS)
            await asyncio.sleep(1.0)

            # Queue for sending to Gemini
            await self.out_queue.put(frame)

        cap.release()

    async def capture_screen(self) -> None:
        """
        Capture screen and track for UI display.

        Same as capture_camera() but for screen capture.
        """
        while not self.should_stop:
            # Use utility function for screen capture
            frame = await asyncio.to_thread(capture_screen_frame)

            if frame is None:
                break

            # Track for UI display
            self.latest_frame = frame
            self.frame_count += 1

            # Wait 1 second (1 FPS)
            await asyncio.sleep(1.0)

            # Queue for sending to Gemini
            await self.out_queue.put(frame)

    # ========================================================================
    # AUDIO INPUT TASK
    # ========================================================================

    async def listen_audio(self) -> None:
        """
        Capture audio from microphone and queue for sending.

        IDENTICAL TO CONSOLE VERSION:
        No UI-specific logic needed here.
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

        kwargs = {"exception_on_overflow": False} if __debug__ else {}

        while not self.should_stop:
            # Read audio chunk
            data = await asyncio.to_thread(
                self.audio_stream.read,
                CHUNK_SIZE,
                **kwargs
            )

            # Queue for sending
            await self.out_queue.put({
                "data": data,
                "mime_type": "audio/pcm"
            })

    # ========================================================================
    # DATA SENDER TASK
    # ========================================================================

    async def send_realtime(self) -> None:
        """
        Send queued data to Gemini in real-time.

        IDENTICAL TO CONSOLE VERSION:
        No UI-specific logic needed here.
        """
        while not self.should_stop:
            msg = await self.out_queue.get()

            mime_type = msg.get("mime_type", "")

            if "audio" in mime_type:
                # Send audio
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=msg["data"],
                        mime_type=mime_type
                    )
                )
            else:
                # Send image/video
                import base64
                image_data = base64.b64decode(msg["data"])

                await self.session.send_realtime_input(
                    media=types.Blob(
                        data=image_data,
                        mime_type=mime_type
                    )
                )

    # ========================================================================
    # RESPONSE RECEIVER TASK (UI-SPECIFIC)
    # ========================================================================

    async def receive_audio(self) -> None:
        """
        Receive responses from Gemini and send to UI.

        UI-SPECIFIC ADDITIONS:
        - Sends transcripts to browser via eel.add_user_transcription()
        - Sends transcripts to browser via eel.add_gemini_transcription()
        - Finalizes messages with frame data
        - Tracks turn timing for frame metadata
        """
        while not self.should_stop:
            try:
                turn = self.session.receive()

                # TURN START: Reset state
                self.current_gemini_transcript = []
                self.frame_count = 0
                self.turn_start_time = time.time()

                # Process response chunks
                async for response in turn:
                    if self.should_stop:
                        break

                    # Audio data
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue

                    # Text response (TEXT mode)
                    if text := response.text:
                        try:
                            eel.add_message("gemini", text)
                        except Exception as e:
                            print(f"Failed to update UI: {e}")
                        print(text, end="")
                        continue

                    # Input transcription (what YOU said)
                    if response.server_content.input_transcription:
                        transcript_text = response.server_content.input_transcription.text
                        self.current_user_transcript.append(transcript_text)

                        # Send to UI in real-time
                        try:
                            eel.add_user_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update user transcript: {e}")

                        print(f"\n[You: {transcript_text}]", end="", flush=True)

                    # Output transcription (what GEMINI is saying)
                    if response.server_content.output_transcription:
                        transcript_text = response.server_content.output_transcription.text
                        self.current_gemini_transcript.append(transcript_text)

                        # Send to UI in real-time
                        try:
                            eel.add_gemini_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update Gemini transcript: {e}")

                        print(transcript_text, end="", flush=True)

                # TURN END: Finalize transcripts

                # Finalize user message
                if self.current_user_transcript:
                    full_user_text = "".join(self.current_user_transcript)
                    try:
                        eel.finalize_user_message(full_user_text)
                    except Exception as e:
                        print(f"Failed to finalize user message: {e}")
                    self.current_user_transcript = []

                # Finalize Gemini message with frame data
                if self.current_gemini_transcript:
                    full_gemini_text = "".join(self.current_gemini_transcript)

                    # Prepare frame data if available
                    frame_data: Optional[Dict[str, Any]] = None

                    if self.video_mode != "none" and self.latest_frame:
                        turn_duration = time.time() - self.turn_start_time if self.turn_start_time else 0

                        frame_data = {
                            "image": self.latest_frame["data"],
                            "mime_type": self.latest_frame["mime_type"],
                            "frame_count": max(1, self.frame_count),
                            "duration": round(turn_duration, 1)
                        }

                    # Send to UI
                    try:
                        eel.finalize_gemini_message(full_gemini_text, frame_data)
                    except Exception as e:
                        print(f"Failed to finalize Gemini message: {e}")

                    self.current_gemini_transcript = []

                # Clear audio queue (handles interruption)
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

            except Exception as e:
                if not self.should_stop:
                    print(f"Error in receive_audio: {e}")
                    traceback.print_exc()
                break

    # ========================================================================
    # AUDIO OUTPUT TASK
    # ========================================================================

    async def play_audio(self) -> None:
        """
        Play audio responses through speakers.

        IDENTICAL TO CONSOLE VERSION:
        No UI-specific logic needed here.
        """
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

    # ========================================================================
    # PLACEHOLDER FOR TEXT INPUT (UI doesn't use this)
    # ========================================================================

    async def wait_for_stop(self) -> None:
        """
        Placeholder task that just waits for stop signal.

        The UI sends text via send_message() instead of console input.
        """
        while not self.should_stop:
            await asyncio.sleep(0.1)


# ============================================================================
# EEL-EXPOSED FUNCTIONS
# ============================================================================

@eel.expose
def start_session(mode: str = DEFAULT_MODE) -> Dict[str, str]:
    """Start a new Gemini Live API session."""
    global audio_loop, is_running, stop_requested

    if is_running:
        return {"status": "error", "message": "Session already running"}

    try:
        print(f"Starting session with mode: {mode}")
        stop_requested = False
        audio_loop = UIAudioLoop(video_mode=mode)

        import threading

        def run_async_session() -> None:
            """Run the async session in a separate thread."""
            global is_running

            time.sleep(0.5)

            try:
                print("Connecting to Gemini...")
                is_running = True
                eel.update_status("Connected - Session started")

                client = genai.Client(http_options={"api_version": "v1beta"})

                async def run_with_config() -> None:
                    """Set up and run all concurrent tasks."""
                    async with (
                        client.aio.live.connect(model=MODEL, config=UI_CONFIG) as session,
                        asyncio.TaskGroup() as tg,
                    ):
                        audio_loop.session = session
                        audio_loop.audio_in_queue = asyncio.Queue()
                        audio_loop.out_queue = asyncio.Queue(maxsize=5)

                        # Create all tasks
                        wait_task = tg.create_task(audio_loop.wait_for_stop())
                        tg.create_task(audio_loop.send_realtime())
                        tg.create_task(audio_loop.listen_audio())

                        # Start video capture if enabled
                        if audio_loop.video_mode == "camera":
                            tg.create_task(audio_loop.capture_camera())
                        elif audio_loop.video_mode == "screen":
                            tg.create_task(audio_loop.capture_screen())

                        tg.create_task(audio_loop.receive_audio())
                        tg.create_task(audio_loop.play_audio())

                        await wait_task

                asyncio.run(run_with_config())

            except Exception as e:
                print(f"Session error: {e}")
                traceback.print_exc()
                try:
                    eel.update_status(f"Error: {str(e)}")
                except:
                    pass
            finally:
                is_running = False
                print("Session ended")
                try:
                    eel.update_status("Disconnected")
                except:
                    pass

        thread = threading.Thread(target=run_async_session, daemon=True)
        thread.start()

        return {"status": "success", "message": "Session started"}

    except Exception as e:
        print(f"Failed to start session: {e}")
        traceback.print_exc()
        is_running = False
        return {"status": "error", "message": str(e)}


@eel.expose
def stop_session() -> Dict[str, str]:
    """Stop the current session."""
    global audio_loop, is_running, stop_requested

    if not is_running:
        return {"status": "error", "message": "No session running"}

    print("Stop requested...")
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
        print(f"Sending message: {text}")

        # Show in UI
        eel.add_message("user", text)

        # Send to Gemini
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
        print(f"Failed to send message: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@eel.expose
def get_status() -> Dict[str, Any]:
    """Get current session status."""
    return {
        "is_running": is_running,
        "mode": audio_loop.video_mode if audio_loop else None
    }


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
    main()