import pyaudio

audio = pyaudio.PyAudio()

try:
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        name = str(info.get("name", ""))

        if "emeet" not in name.lower() and "m0" not in name.lower():
            continue

        print(f"Device index: {index}")
        print(f"Name: {name}")
        print(f"Input channels: {info.get('maxInputChannels')}")
        print(f"Output channels: {info.get('maxOutputChannels')}")
        print(f"Default sample rate: {info.get('defaultSampleRate')}")
        print()
finally:
    audio.terminate()