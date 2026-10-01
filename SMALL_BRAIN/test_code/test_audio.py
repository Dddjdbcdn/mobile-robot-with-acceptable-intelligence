import pyaudio

audio = pyaudio.PyAudio()

try:
    default_input = int(audio.get_default_input_device_info()["index"])
    default_output = int(audio.get_default_output_device_info()["index"])
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        defaults = []
        if index == default_input:
            defaults.append("default input")
        if index == default_output:
            defaults.append("default output")
        suffix = f" ({', '.join(defaults)})" if defaults else ""
        print(
            f"{index}: {info.get('name')}{suffix}; "
            f"input={info.get('maxInputChannels')}; "
            f"output={info.get('maxOutputChannels')}; "
            f"rate={info.get('defaultSampleRate')}"
        )
finally:
    audio.terminate()
