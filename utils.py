"""
Gemini Live API - Shared Utility Functions

This module contains reusable functions for video/screen capture that can be
shared between the console app (main.py) and the web UI (app_ui.py).

DESIGN PRINCIPLE:
Instead of inheritance, we use composition - both apps import these utilities
and use them as needed. This makes the code more maintainable and testable.
"""

import base64
import io
from typing import Optional, Dict

import cv2
import PIL.Image
import mss
import mss.tools


# ============================================================================
# CAMERA CAPTURE UTILITIES
# ============================================================================

def capture_camera_frame(cap: cv2.VideoCapture) -> Optional[Dict[str, str]]:
    """
    Capture a single frame from a camera and encode it for Gemini.

    This is a SYNCHRONOUS function - it blocks while capturing.
    Use asyncio.to_thread() when calling from async code.

    Args:
        cap: OpenCV VideoCapture object (already opened)

    Returns:
        Dictionary with:
        - "mime_type": "image/jpeg"
        - "data": Base64-encoded JPEG bytes

        Returns None if capture fails.

    PROCESSING STEPS:
    1. Read frame from camera (BGR color space)
    2. Convert BGR -> RGB (PIL/Gemini expect RGB)
    3. Resize to max 1024x1024 (reduce bandwidth)
    4. Encode as JPEG
    5. Base64 encode for transmission

    GEMINI REQUIREMENTS:
    - Accepts JPEG or PNG
    - Recommends max 1024x1024 for efficiency
    - Processes video at ~1 FPS
    """
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        # Capture failed (camera disconnected, busy, etc.)
        return None

    # Convert BGR (OpenCV default) to RGB (expected by PIL/Gemini)
    # Without this conversion, images would have a blue tint
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create PIL Image from numpy array
    img = PIL.Image.fromarray(frame_rgb)

    # Resize to fit within 1024x1024 (maintains aspect ratio)
    img.thumbnail((1024, 1024))

    # Encode as JPEG in memory
    image_io = io.BytesIO()
    img.save(image_io, format="jpeg")
    image_io.seek(0)

    # Read bytes and base64 encode
    image_bytes = image_io.read()

    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(image_bytes).decode()
    }


# ============================================================================
# SCREEN CAPTURE UTILITIES
# ============================================================================

def capture_screen_frame() -> Optional[Dict[str, str]]:
    """
    Capture the entire screen and encode it for Gemini.

    This is a SYNCHRONOUS function - it blocks while capturing.
    Use asyncio.to_thread() when calling from async code.

    Returns:
        Dictionary with:
        - "mime_type": "image/jpeg"
        - "data": Base64-encoded JPEG bytes

        Returns None if capture fails.

    SCREEN CAPTURE:
    Uses mss library which is:
    - Fast (uses native OS APIs)
    - Cross-platform (Windows, Mac, Linux)
    - Captures at monitor's native resolution

    USE CASES:
    - Screen sharing
    - Showing Gemini what you're looking at
    - Debugging visual issues
    - Presentations
    """
    try:
        # Initialize screen capture
        sct = mss.mss()

        # monitors[0] is the entire virtual screen (all monitors combined)
        # monitors[1], [2], etc. are individual monitors
        monitor = sct.monitors[0]

        # Grab the screen
        screenshot = sct.grab(monitor)

        # Convert to PNG bytes
        png_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)

        # Open PNG with PIL
        img = PIL.Image.open(io.BytesIO(png_bytes))

        # Re-encode as JPEG (smaller file size)
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        # Read bytes and base64 encode
        image_bytes = image_io.read()

        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_bytes).decode()
        }

    except Exception as e:
        print(f"Screen capture failed: {e}")
        return None


# ============================================================================
# AUDIO FORMAT CONSTANTS
# ============================================================================
# These are used by both apps for PyAudio configuration

import pyaudio

# Audio format constants for PyAudio
# These match the requirements of Gemini Live API
FORMAT = pyaudio.paInt16  # 16-bit PCM (Pulse Code Modulation)
CHANNELS = 1  # Mono audio (single channel)
SEND_SAMPLE_RATE = 16000  # Input: 16kHz (required by Live API)
RECEIVE_SAMPLE_RATE = 24000  # Output: 24kHz (Live API always outputs at this rate)
CHUNK_SIZE = 1024  # Audio frames per buffer (affects latency)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_microphone_info() -> Dict:
    """
    Get information about the system's default microphone.

    Returns:
        Dict with microphone properties (device index, name, etc.)
    """
    pya = pyaudio.PyAudio()
    mic_info = pya.get_default_input_device_info()
    pya.terminate()
    return mic_info