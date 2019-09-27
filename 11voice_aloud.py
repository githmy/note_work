def voice():
    import win32com.client

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak("Hello, it works!")
    speaker.Speak("你好，郭璞!")

    import pyttsx3

    engine = pyttsx3.init()
    # 音量
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume - 0.25)
    # 语速
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 50)
    # 更换发音人声音
    voices = engine.getProperty("voices")
    for voice in voices:
        engine.setProperty('voice', voice.id)
        print(voice.id, voice.languages, voice.age, voice.gender, voice.name)
        engine.say("你好，郭璞!")
        engine.runAndWait()
    engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
    engine.say("Good")
    engine.say("你好，郭璞!")
    engine.runAndWait()


if __name__ == '__main__':
    voice()
