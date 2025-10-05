"""
Gemini Live API - Web UI Application using Eel

This provides a web-based interface for the Gemini Live API application.
It imports the AudioLoop from main.py and provides a graphical interface
to control the audio/video streaming session.

## Setup

Install additional dependencies:
```
pip install eel
```

## Run

```
python app_ui.py
```

This will open a web browser with the UI.
"""

import asyncio
import eel
import sys
import traceback
from pathlib import Path
from typing import Optional

# Import the AudioLoop class from main.py
from main import AudioLoop, DEFAULT_MODE

# Initialize Eel with the web folder
# We'll create the web files in a 'web' directory
eel.init('web')

# Global state
audio_loop: Optional[AudioLoop] = None
session_task: Optional[asyncio.Task] = None
is_running = False
stop_requested = False  # Flag to request stopping the session


class UIAudioLoop(AudioLoop):
    """
    Extended AudioLoop that sends updates to the UI.

    Overrides methods to send status updates and transcripts to the browser.
    """

    def __init__(self, video_mode: str = DEFAULT_MODE) -> None:
        super().__init__(video_mode)
        self.should_stop = False  # Flag to signal stopping

    async def send_text(self) -> None:
        """
        Modified version that receives text from the UI instead of console.

        This version doesn't use input() but waits for messages from the UI.
        Also checks for stop signals.
        """
        while not self.should_stop:
            # Wait a bit to keep the task alive
            # Actual text sending is handled by send_message_from_ui()
            await asyncio.sleep(0.1)

    async def receive_audio(self) -> None:
        """
        Modified to send transcripts to the UI.
        """
        while not self.should_stop:
            try:
                turn = self.session.receive()

                async for response in turn:
                    if self.should_stop:
                        break
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        # Send transcript to UI
                        try:
                            eel.update_transcript(text)
                        except Exception as e:
                            print(f"Failed to update transcript in UI: {e}")
                        print(text, end="")

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
    """
    Start a Gemini Live session with the specified video mode.

    Args:
        mode: One of "camera", "screen", or "none"
    """
    global audio_loop, session_task, is_running, stop_requested

    if is_running:
        print("ERROR: Session already running")
        return {"status": "error", "message": "Session already running"}

    try:
        print(f"Starting session with mode: {mode}")

        # Reset stop flag
        stop_requested = False

        # Create the audio loop
        audio_loop = UIAudioLoop(video_mode=mode)

        # Run the session in the asyncio event loop
        # We need to run this in a separate thread since Eel has its own event loop
        import threading
        import time

        def run_async_session():
            global is_running

            # Give the UI a moment to be ready before sending status updates
            time.sleep(0.5)

            try:
                print("Session thread started, connecting to Gemini...")
                is_running = True  # Set this only when actually starting
                eel.update_status("Connected - Session started")

                asyncio.run(audio_loop.run())

            except Exception as e:
                print(f"Session error: {e}")
                traceback.print_exc()
                try:
                    eel.update_status(f"Error: {str(e)}")
                except:
                    print("Failed to update UI with error status")
            finally:
                is_running = False
                print("Session ended")
                try:
                    eel.update_status("Disconnected")
                except:
                    print("Failed to update UI with disconnected status")

        thread = threading.Thread(target=run_async_session, daemon=True)
        thread.start()

        print("Session thread launched successfully")
        return {"status": "success", "message": "Session started"}

    except Exception as e:
        print(f"Failed to start session: {e}")
        traceback.print_exc()
        is_running = False
        return {"status": "error", "message": str(e)}


@eel.expose
def stop_session():
    """
    Stop the current Gemini Live session.
    """
    global audio_loop, is_running, stop_requested

    if not is_running:
        print("ERROR: No session running")
        return {"status": "error", "message": "No session running"}

    print("Stop requested - signaling session to stop...")
    stop_requested = True

    # THIS IS MISSING:
    try:
        eel.update_status("Stopping...")
    except:
        print("Failed to update status to 'Stopping...'")

    if audio_loop:
        # Signal the audio loop to stop
        audio_loop.should_stop = True
        print("Stop signal sent to audio loop")

    return {"status": "success", "message": "Stopping session..."}


@eel.expose
def send_message(text: str):
    """
    Send a text message to Gemini from the UI.

    Args:
        text: The message to send

    Note: This must be a regular function, not async, because Eel calls it from JavaScript.
    We create a new event loop to run the async code.
    """
    global audio_loop, is_running

    if not is_running or not audio_loop or not audio_loop.session:
        print("ERROR: No active session")
        return {"status": "error", "message": "No active session"}

    try:
        print(f"Sending message: {text}")
        from google.genai import types

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async send operation
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

        # Echo user message to transcript
        eel.update_transcript(f"\n\nYou: {text}\n\nGemini: ")

        print("Message sent successfully")
        return {"status": "success"}
    except Exception as e:
        print(f"Failed to send message: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@eel.expose
def get_status():
    """
    Get the current session status.
    """
    return {
        "is_running": is_running,
        "mode": audio_loop.video_mode if audio_loop else None
    }


def main():
    """
    Main entry point for the UI application.
    """
    # Check if web folder exists
    web_dir = Path(__file__).parent / 'web'
    if not web_dir.exists():
        print("Creating web directory and files...")
        create_web_files()

    # Start the Eel application
    # Opens in default browser
    try:
        eel.start('index.html', size=(1000, 800), mode='default')
    except (SystemExit, KeyboardInterrupt):
        print("\nShutting down...")
        sys.exit(0)


def create_web_files():
    """
    Create the web UI files if they don't exist.
    """
    web_dir = Path(__file__).parent / 'web'
    web_dir.mkdir(exist_ok=True)

    # Read template files from web_templates directory
    templates_dir = Path(__file__).parent / 'web_templates'

    if not templates_dir.exists():
        print(f"Error: web_templates directory not found at {templates_dir}")
        print("Please create the web_templates directory with index.html, style.css, and script.js")
        sys.exit(1)

    # Copy files from templates to web directory
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