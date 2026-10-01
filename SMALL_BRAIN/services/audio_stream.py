import asyncio
import base64
import json
import os
from pathlib import Path
import queue
import threading
import numpy as np
import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

class AudioApp:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.play_queue = queue.Queue()
        self.mic_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()

        self.input_device_index = self._resolve_device(
            "AUDIO_INPUT_DEVICE", input_device=True
        )
        self.output_device_index = self._resolve_device(
            "AUDIO_OUTPUT_DEVICE", input_device=False
        )
        self._playback_error_reported = False
        self._print_device("input", self.input_device_index)
        self._print_device("output", self.output_device_index)

        self.input_stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, 
            input=True, frames_per_buffer=CHUNK,
            input_device_index=self.input_device_index,
            stream_callback=self.mic_callback
        )
        
        # Output Stream (Speaker)
        self.output_stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, 
            output=True, frames_per_buffer=CHUNK,
            output_device_index=self.output_device_index,
        )
        
        # Dedicated thread for blocking audio playback
        self.running = True
        self.play_thread = threading.Thread(
            target=self.playback_worker,
            name="audio-playback",
            daemon=True,
        )
        self.play_thread.start()

    def _resolve_device(self, environment_name, *, input_device):
        configured = os.environ.get(environment_name)
        if configured is None or not configured.strip():
            getter = (
                self.p.get_default_input_device_info
                if input_device
                else self.p.get_default_output_device_info
            )
            return int(getter()["index"])

        configured = configured.strip()
        try:
            index = int(configured)
        except ValueError:
            matches = []
            for index in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(index)
                channels = info.get(
                    "maxInputChannels" if input_device else "maxOutputChannels",
                    0,
                )
                if channels and configured.casefold() in str(
                    info.get("name", "")
                ).casefold():
                    matches.append(index)
            if len(matches) != 1:
                direction = "input" if input_device else "output"
                raise RuntimeError(
                    f"{environment_name}={configured!r} matched "
                    f"{len(matches)} {direction} devices; use an exact index"
                )
            index = matches[0]

        info = self.p.get_device_info_by_index(index)
        channel_key = "maxInputChannels" if input_device else "maxOutputChannels"
        if int(info.get(channel_key) or 0) < CHANNELS:
            direction = "input" if input_device else "output"
            raise RuntimeError(
                f"Audio device {index} ({info.get('name')}) has no {direction} channel"
            )
        return index

    def _print_device(self, direction, index):
        info = self.p.get_device_info_by_index(index)
        print(
            f"[Audio {direction}] index={index} name={info.get('name')!r} "
            f"default_rate={info.get('defaultSampleRate')}"
        )

    def mic_callback(self, in_data, frame_count, time_info, status):
        # Safely push raw mic bytes to the async event loop
        self.loop.call_soon_threadsafe(self.mic_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    def playback_worker(self):
        speed_factor = 1.25
        
        while self.running:
            try:
                data = self.play_queue.get(timeout=0.1)
                
                # 1. Convert bytes to numpy array (16-bit PCM)
                audio_array = np.frombuffer(data, dtype=np.int16)
                
                # 2. Resample by picking indices (The "lowest effort" trick)
                # This creates the higher pitch/faster speed instantly
                indices = np.round(np.arange(0, len(audio_array), speed_factor)).astype(int)
                indices = indices[indices < len(audio_array)]
                resampled_data = audio_array[indices]
                
                # 3. Write back to stream
                self.output_stream.write(resampled_data.tobytes())
                self._playback_error_reported = False
                
            except queue.Empty:
                continue
            except Exception as error:
                if not self._playback_error_reported:
                    print(
                        "[Audio playback error] "
                        f"{type(error).__name__}: {error}"
                    )
                    self._playback_error_reported = True

    def stop(self, join_timeout=2.0):
        self.running = False
        self.clear_queue()

        # Stop new callbacks first, then give an in-progress speaker write a
        # bounded amount of time to finish. The daemon flag is the final guard
        # against a faulty audio backend preventing interpreter shutdown.
        if self.input_stream.is_active():
            self.input_stream.stop_stream()
        self.play_thread.join(timeout=join_timeout)
        if self.play_thread.is_alive() and self.output_stream.is_active():
            self.output_stream.stop_stream()
            self.play_thread.join(timeout=join_timeout)

        self.input_stream.close()
        if self.output_stream.is_active():
            self.output_stream.stop_stream()
        self.output_stream.close()
        self.p.terminate()

    
    def clear_queue(self):
        """Instantly empty the playback queue when the user interrupts."""
        while not self.play_queue.empty():
            try:
                self.play_queue.get_nowait()
            except queue.Empty:
                break

# 5. Core Asynchronous Logic
async def send_mic_audio(ws, app):
    """Continuously stream microphone data to OpenAI."""
    while True:
        data = await app.mic_queue.get()
        base64_audio = base64.b64encode(data).decode('utf-8')
        event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        await ws.send(json.dumps(event))
