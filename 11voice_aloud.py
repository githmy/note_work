def voice():
    import win32com.client

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak("Hello, it works!")
    speaker.Speak("你好，郭璞!")

    import pyttsx3

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    for item in voices:
        print(item.id, item.languages)
    engine.setProperty("voice", "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0")
    engine.say("Good")
    engine.say("你好，郭璞!")
    engine.runAndWait()


if __name__ == '__main__':
    voice()
