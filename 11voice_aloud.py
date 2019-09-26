def voice():
    import win32com.client

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak("Hello, it works!")
    speaker.Speak("你好，郭璞!")


if __name__ == '__main__':
    voice()
