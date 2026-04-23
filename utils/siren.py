import os
import time
import threading
from playsound import playsound

last_alert_time = 0


def _play_siren_background(path: str) -> None:
    """Run blocking audio playback in a background thread."""
    try:
        playsound(path)
    except Exception as e:
        # Avoid crashing main loop; just log once
        print("SIREN ERROR:", e)


def play_siren():

    global last_alert_time

    current_time = time.time()

    # Prevent siren spam
    if current_time - last_alert_time < 3:
        return

    last_alert_time = current_time

    siren_path = os.path.join("assets", "siren.mp3")

    # Start non-blocking background thread so main loop never freezes
    t = threading.Thread(target=_play_siren_background, args=(siren_path,))
    t.daemon = True
    t.start()