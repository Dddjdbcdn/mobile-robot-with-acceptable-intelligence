import asyncio
import websockets
import json
import base64
import os
import sys
import threading
import queue
import pyaudio

# 1. Credentials
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    sys.exit(1)

URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-mini"

# 2. Audio Constants (OpenAI expects 24kHz, mono, 16-bit PCM)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000
CHUNK = 1024

# 3. Define our Basic Tool (Function Call)
def check_device_status(device_name: str) -> str:
    """A mock tool to check system/device status."""
    print(f"\n[🔧 Tool Executing] Checking status for '{device_name}'...")
    return f"The {device_name} is fully operational."

TOOLS = [
    {
        "type": "function",
        "name": "check_device_status",
        "description": "Check the status of a specific device or hardware system.",
        "parameters": {
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "The name of the device, like 'speakerphone' or 'Asus NUC'."
                }
            },
            "required": ["device_name"]
        }
    }
]

# 4. Hardware Audio Controller
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
        # Continuously play audio chunks arriving from OpenAI
        while self.running:
            try:
                data = self.play_queue.get(timeout=0.1)
                self.output_stream.write(data)
            except queue.Empty:
                continue
            except Exception as e:
                # 👉 If PyAudio crashes, it will print here instead of dying silently
                print(f"\n❌ PLAYBACK THREAD ERROR: {e}")

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

async def handle_tool_call(ws, function_name, arguments, call_id):
    """Execute the local python tool and send the result back to the model."""
    if function_name == "check_device_status":
        args = json.loads(arguments)
        result = check_device_status(args.get("device_name"))
        
        # 1. Send the result of the tool back
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result
            }
        }
        await ws.send(json.dumps(event))
        
        # 2. Tell the model to synthesize a voice response based on the tool result
        await ws.send(json.dumps({"type": "response.create"}))

async def receive_events(ws, app):
    async for message in ws:
        event = json.loads(message)
        event_type = event.get("type")

        if event_type == "error":
            print(f"\n❌ OPENAI ERROR: {json.dumps(event.get('error'), indent=2)}")

        # 👉 ADD THIS: Instantly stop audio when you interrupt!
        elif event_type == "input_audio_buffer.speech_started":
            print("\n🛑 [Barge-in detected! Stopping audio...]")
            app.clear_queue()
            
        # 🔊 Play AI Voice (FIXED EVENT NAME)
        elif event_type == "response.output_audio.delta":
            audio_base64 = event.get("delta")
            if audio_base64:
                app.play_queue.put(base64.b64decode(audio_base64))
        
        # 📝 Print live transcript (FIXED EVENT NAME)
        elif event_type == "response.output_audio_transcript.delta":
             print(event.get("delta", ""), end="", flush=True)
             
        elif event_type == "response.output_audio_transcript.done":
             print("\n")
             
        # 🔧 Handle the model deciding to call our tool
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            func_name = event.get("name")
            args = event.get("arguments")
            await handle_tool_call(ws, func_name, args, call_id)

async def main():
    headers = {
        "Authorization": "Bearer " + OPENAI_API_KEY,
        "OpenAI-Safety-Identifier": "hashed-user-id",
    }


    print("Initializing Audio. (Ensure your Speakerphone is the default system device)...")
    app = AudioApp()

    print("Connecting to OpenAI Realtime API...")
    try:
        async with websockets.connect(URL, additional_headers=headers) as ws:
            print("\n✅ Connected! Just start speaking to the agent.\n")
            
            # Configure Session: Attach tools and enable auto-turn detection
            # Configure Session: Attach tools and enable auto-turn detection
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000,
                            },
                            "turn_detection": {
                                "type": "server_vad"
                            }
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000,
                            },
                            "voice": "marin",
                        }
                    },
                    
                    "instructions": "You are a helpful AI assistant running on an Asus NUC N97. Keep responses natural and conversational. You have access to a tool to check device status.",
                    "tools": TOOLS 
                }
            }
            await ws.send(json.dumps(session_update))

            # Run microphone streaming and server receiving concurrently
            await asyncio.gather(
                send_mic_audio(ws, app),
                receive_events(ws, app)
            )
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed by server.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up audio hardware...")
        app.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")