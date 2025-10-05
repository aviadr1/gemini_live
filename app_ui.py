"""
Gemini Live API - Web UI Application with Full Transcription Support

This version captures and displays:
1. Your voice input (transcribed automatically)
2. Gemini's voice output (transcribed automatically)
3. Text messages you type
"""

import asyncio
import eel
import sys
import traceback
from pathlib import Path
from typing import Optional

# Import the AudioLoop class and types
from main import AudioLoop, DEFAULT_MODE
from google.genai import types
from google.genai.types import LiveConnectConfig, Modality

# Initialize Eel
eel.init('web')

# Global state
audio_loop: Optional[AudioLoop] = None
session_task: Optional[asyncio.Task] = None
is_running = False
stop_requested = False

# Updated CONFIG with both input and output transcription
UI_CONFIG = types.LiveConnectConfig(
    response_modalities=[Modality.AUDIO],
    input_audio_transcription=types.AudioTranscriptionConfig(),  # Transcribes YOUR voice
    output_audio_transcription=types.AudioTranscriptionConfig()  # Transcribes Gemini's voice
)


class UIAudioLoop(AudioLoop):
    """
    Extended AudioLoop with transcription support for the UI.
    """

    def __init__(self, video_mode: str = DEFAULT_MODE) -> None:
        super().__init__(video_mode)
        self.should_stop = False
        self.current_user_transcript = []  # Accumulate user transcription chunks
        self.current_gemini_transcript = []  # Accumulate Gemini transcription chunks

    async def send_text(self) -> None:
        """Modified to wait for UI messages instead of console input."""
        while not self.should_stop:
            await asyncio.sleep(0.1)

    async def receive_audio(self) -> None:
        """
        Modified to handle and display all transcriptions.
        """
        while not self.should_stop:
            try:
                turn = self.session.receive()

                # Reset Gemini transcript at start of new turn
                self.current_gemini_transcript = []

                async for response in turn:
                    if self.should_stop:
                        break

                    # Handle audio data
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue

                    # Handle text responses (for TEXT mode)
                    if text := response.text:
                        try:
                            eel.add_message("gemini", text)
                        except Exception as e:
                            print(f"Failed to update UI: {e}")
                        print(text, end="")
                        continue

                    # NEW: Handle input transcription (what YOU said)
                    if response.server_content.input_transcription:
                        transcript_text = response.server_content.input_transcription.text
                        self.current_user_transcript.append(transcript_text)

                        # Send to UI incrementally
                        try:
                            eel.add_user_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update user transcript in UI: {e}")

                        print(f"\n[You: {transcript_text}]", end="", flush=True)

                    # NEW: Handle output transcription (what Gemini is saying)
                    if response.server_content.output_transcription:
                        transcript_text = response.server_content.output_transcription.text
                        self.current_gemini_transcript.append(transcript_text)

                        # Send to UI incrementally
                        try:
                            eel.add_gemini_transcription(transcript_text)
                        except Exception as e:
                            print(f"Failed to update Gemini transcript in UI: {e}")

                        print(transcript_text, end="", flush=True)

                # At end of turn, finalize the transcripts
                if self.current_user_transcript:
                    full_user_text = "".join(self.current_user_transcript)
                    try:
                        eel.finalize_user_message(full_user_text)
                    except Exception as e:
                        print(f"Failed to finalize user message: {e}")
                    self.current_user_transcript = []

                if self.current_gemini_transcript:
                    full_gemini_text = "".join(self.current_gemini_transcript)
                    try:
                        eel.finalize_gemini_message(full_gemini_text)
                    except Exception as e:
                        print(f"Failed to finalize Gemini message: {e}")
                    self.current_gemini_transcript = []

                # Clear audio queue on interruption
                while not self.audio_in_queue.empty():
                    self.audio_in_queue.get_nowait()

            except Exception as e:
                if not self.should_stop:
                    print(f"Error in receive_audio: {e}")
                    traceback.print_exc()
                break


@eel.expose
def start_session(mode: str = DEFAULT_MODE):
    """Start a Gemini Live session."""
    global audio_loop, session_task, is_running, stop_requested

    if is_running:
        return {"status": "error", "message": "Session already running"}

    try:
        print(f"Starting session with mode: {mode}")
        stop_requested = False
        audio_loop = UIAudioLoop(video_mode=mode)

        import threading
        import time

        def run_async_session():
            global is_running

            time.sleep(0.5)

            try:
                print("Connecting to Gemini...")
                is_running = True
                eel.update_status("Connected - Session started")

                # Use the UI_CONFIG with transcription enabled
                from google import genai
                client = genai.Client(http_options={"api_version": "v1beta"})

                # Create a custom run method with our config
                async def run_with_config():
                    async with (
                        client.aio.live.connect(model="models/gemini-2.0-flash-live-001", config=UI_CONFIG) as session,
                        asyncio.TaskGroup() as tg,
                    ):
                        audio_loop.session = session
                        audio_loop.audio_in_queue = asyncio.Queue()
                        audio_loop.out_queue = asyncio.Queue(maxsize=5)

                        send_text_task = tg.create_task(audio_loop.send_text())
                        tg.create_task(audio_loop.send_realtime())
                        tg.create_task(audio_loop.listen_audio())

                        if audio_loop.video_mode == "camera":
                            tg.create_task(audio_loop.get_frames())
                        elif audio_loop.video_mode == "screen":
                            tg.create_task(audio_loop.get_screen())

                        tg.create_task(audio_loop.receive_audio())
                        tg.create_task(audio_loop.play_audio())

                        await send_text_task

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
def stop_session():
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
def send_message(text: str):
    """Send a text message to Gemini."""
    global audio_loop, is_running

    if not is_running or not audio_loop or not audio_loop.session:
        return {"status": "error", "message": "No active session"}

    try:
        print(f"Sending message: {text}")

        # Echo user message to transcript first
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
def get_status():
    """Get current session status."""
    return {
        "is_running": is_running,
        "mode": audio_loop.video_mode if audio_loop else None
    }


def main():
    """Main entry point."""
    web_dir = Path(__file__).parent / 'web'
    if not web_dir.exists():
        print("Creating web directory and files...")
        create_web_files()

    try:
        eel.start('index.html', size=(1000, 800), mode='default')
    except (SystemExit, KeyboardInterrupt):
        print("\nShutting down...")
        sys.exit(0)


def create_web_files():
    """Create web UI files."""
    web_dir = Path(__file__).parent / 'web'
    web_dir.mkdir(exist_ok=True)

    templates_dir = Path(__file__).parent / 'web_templates'

    if not templates_dir.exists():
        print(f"Error: web_templates directory not found at {templates_dir}")
        sys.exit(1)

    for filename in ['index.html', 'style.css', 'script.js']:
        src = templates_dir / filename
        dst = web_dir / filename

        if not src.exists():
            print(f"Error: {filename} not found in web_templates/")
            sys.exit(1)

        dst.write_text(src.read_text())

    print(f"✓ Created web files in {web_dir}")


if __name__ == "__main__":
    main()