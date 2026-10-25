import asyncio
import queue
import sys
import threading
import time
from types import SimpleNamespace
import unittest

sys.modules.setdefault(
    "pyaudio",
    SimpleNamespace(paInt16=8, paContinue=0),
)

from services.stream.audio_stream import AudioApp


class AudioControlTests(unittest.TestCase):
    def setUp(self):
        # Exercise the state/queue logic without opening real audio devices.
        self.app = AudioApp.__new__(AudioApp)
        self.app.play_queue = queue.Queue()
        self.app.mic_queue = asyncio.Queue()
        self.app.output_muted = False
        self.app.input_muted = False
        self.app._voice_lock = threading.Lock()
        self.app._person_speaking = False
        self.app._playback_active = False
        self.app._last_playback_at = 0.0

    def test_muting_output_clears_and_blocks_playback_queue(self):
        self.app.queue_audio(b"before")
        self.app.set_output_muted(True)
        self.app.queue_audio(b"during")

        self.assertTrue(self.app.output_muted)
        self.assertTrue(self.app.play_queue.empty())

        self.app.set_output_muted(False)
        self.app.queue_audio(b"after")
        self.assertEqual(self.app.play_queue.get_nowait(), b"after")

    def test_covering_ear_clears_mic_and_queues_server_clear_marker(self):
        self.app.mic_queue.put_nowait(b"noise")

        self.app.set_input_muted(True)

        self.assertTrue(self.app.input_muted)
        self.assertIsNone(self.app.mic_queue.get_nowait())
        self.assertTrue(self.app.mic_queue.empty())

    def test_voice_state_prefers_detected_person_speech(self):
        self.app._playback_active = True
        self.app.set_person_speaking(True)

        self.assertEqual(self.app.voice_state(), "listening")

    def test_voice_state_tracks_recent_speaker_playback(self):
        self.app._last_playback_at = time.monotonic()
        self.assertEqual(self.app.voice_state(), "talking")

        self.app.set_output_muted(True)
        self.assertEqual(self.app.voice_state(), "idle")


if __name__ == "__main__":
    unittest.main()
