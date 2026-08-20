import asyncio
import base64
import json
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

        self.input_stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, 
            input=True, frames_per_buffer=CHUNK,
            stream_callback=self.mic_callback
        )
        
        # Output Stream (Speaker)
        self.output_stream = self.p.open(
            format=FORMAT, channels=CHANNELS, rate=RATE, 
            output=True, frames_per_buffer=CHUNK
        )
        
        # Dedicated thread for blocking audio playback
        self.running = True
        self.play_thread = threading.Thread(target=self.playback_worker)
        self.play_thread.start()

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
                
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        self.play_thread.join()
        self.input_stream.stop_stream()
        self.input_stream.close()
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