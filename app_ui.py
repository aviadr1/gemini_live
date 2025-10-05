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
NEW FEATURES:
- Configurable frame rate (0.5 - 5 FPS)
- Video narration mode for continuous AI feedback
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
    Standalone audio loop for web UI with configurable frame rate and narration mode.
    """

    def __init__(
        self,
        video_mode: str = DEFAULT_MODE,
        fps: float = DEFAULT_FPS,
        narration_mode: bool = False,
        narration_interval: float = DEFAULT_NARRATION_INTERVAL
    ) -> None:
        """
        Initialize the UI audio loop.

        Args:
            video_mode: "camera", "screen", or "none"
            fps: Frames per second (0.5 to 5.0)
            narration_mode: Enable automatic video narration
            narration_interval: Seconds between narration prompts
        """
        # Configuration
        self.video_mode: str = video_mode
        self.fps: float = max(0.5, min(5.0, fps))  # Clamp between 0.5 and 5
        self.frame_interval: float = 1.0 / self.fps
        self.narration_mode: bool = narration_mode
        self.narration_interval: float = narration_interval
        self.should_stop: bool = False

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

    # ========================================================================
    # VIDEO CAPTURE TASKS
    # ========================================================================

    async def capture_camera(self) -> None:
        """Capture frames from webcam at configured FPS."""
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)

        while not self.should_stop:
            frame = await asyncio.to_thread(capture_camera_frame, cap)

            if frame is None:
                break

            self.latest_frame = frame
            self.frame_count += 1

            # Use configurable frame interval
            await asyncio.sleep(self.frame_interval)
            await self.out_queue.put(frame)

        cap.release()

    async def capture_screen(self) -> None:
        """Capture screen at configured FPS."""
        while not self.should_stop:
            frame = await asyncio.to_thread(capture_screen_frame)

            if frame is None:
                break

            self.latest_frame = frame
            self.frame_count += 1

            # Use configurable frame interval
            await asyncio.sleep(self.frame_interval)
            await self.out_queue.put(frame)

    # ========================================================================
    # VIDEO NARRATION TASK
    # ========================================================================

    async def video_narrator(self) -> None:
        """
        Periodically send prompts to get AI narration of video content.

        This is useful for:
        - Fitness/yoga form feedback
        - Cooking demonstrations
        - Art/craft tutorials
        - Any scenario where you want continuous AI commentary
        """
        narration_prompts = [
            "What do you see? Provide brief feedback.",
            "Describe what's happening and offer any suggestions.",
            "Analyze the current activity and provide guidance.",
        ]

        prompt_index = 0

        while not self.should_stop:
            await asyncio.sleep(self.narration_interval)

            if self.should_stop:
                break

            current_time = time.time()

            # Only send if enough time has passed since last narration
            if current_time - self.last_narration_time >= self.narration_interval:
                try:
                    prompt = narration_prompts[prompt_index % len(narration_prompts)]

                    await self.session.send_client_content(
                        turns=types.Content(
                            role="user",
                            parts=[types.Part(text=prompt)]
                        ),
                        turn_complete=True
                    )

                    self.last_narration_time = current_time
                    prompt_index += 1

                    print(f"[Narration prompt sent: {prompt}]")

                except Exception as e:
                    print(f"Failed to send narration prompt: {e}")

    # ========================================================================
    # AUDIO INPUT TASK
    # ========================================================================

    async def listen_audio(self) -> None:
        """Capture audio from microphone and queue for sending."""
        mic_info = pya.get_default_input_device_info()

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

            await self.out_queue.put({
                "data": data,
                "mime_type": "audio/pcm"
            })

    # ========================================================================
    # DATA SENDER TASK
    # ========================================================================

    async def send_realtime(self) -> None:
        """Send queued data to Gemini in real-time."""
        while not self.should_stop:
            msg = await self.out_queue.get()

            mime_type = msg.get("mime_type", "")

            if "audio" in mime_type:
                await self.session.send_realtime_input(
                    audio=types.Blob(
                        data=msg["data"],
                        mime_type=mime_type
                    )
                )
            else:
                import base64
                image_data = base64.b64decode(msg["data"])

                await self.session.send_realtime_input(
                    media=types.Blob(
                        data=image_data,
                        mime_type=mime_type
                    )
                )

    # ========================================================================
    # RESPONSE RECEIVER TASK
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

                self.current_gemini_transcript = []
                self.frame_count = 0
                self.turn_start_time = time.time()

                async for response in turn:
                    if self.should_stop:
                        break

                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue

                    if text := response.text:
                        try:
                            eel.add_message("gemini", text)
                        except Exception as e:
                            print(f"Failed to update UI: {e}")
                        print(text, end="")
                        continue

                    if response.server_content.input_transcription:
                        transcript_text = response.server_content.input_transcription.text
                        self.current_user_transcript.append(transcript_text)

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

                # TURN END: Finalize transcripts

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

                    try:
                        eel.finalize_gemini_message(full_gemini_text, frame_data)
                    except Exception as e:
                        print(f"Failed to finalize Gemini message: {e}")

                    self.current_gemini_transcript = []

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
        """Play audio responses through speakers."""
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
    narration_interval: float = DEFAULT_NARRATION_INTERVAL
) -> Dict[str, str]:
    """Start a new Gemini Live API session with configuration."""
    global audio_loop, is_running, stop_requested

    if is_running:
        return {"status": "error", "message": "Session already running"}

    try:
        print(f"Starting session: mode={mode}, fps={fps}, narration={narration_mode}")
        stop_requested = False
        audio_loop = UIAudioLoop(
            video_mode=mode,
            fps=fps,
            narration_mode=narration_mode,
            narration_interval=narration_interval
        )

        import threading

        def run_async_session() -> None:
            global is_running

            time.sleep(0.5)

            try:
                print("Connecting to Gemini...")
                is_running = True

                status_msg = f"Connected - {mode.title()} @ {fps} FPS"
                if narration_mode:
                    status_msg += f" | Narration every {narration_interval}s"
                eel.update_status(status_msg)

                client = genai.Client(http_options={"api_version": "v1beta"})

                async def run_with_config() -> None:
                    async with (
                        client.aio.live.connect(model=MODEL, config=UI_CONFIG) as session,
                        asyncio.TaskGroup() as tg,
                    ):
                        audio_loop.session = session
                        audio_loop.audio_in_queue = asyncio.Queue()
                        audio_loop.out_queue = asyncio.Queue(maxsize=20)

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
    load_dotenv()
    main()