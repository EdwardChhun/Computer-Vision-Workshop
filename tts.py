# !pip install pyttsx3

import pyttsx3

engine = pyttsx3.init() 

def speak(text):
        engine.say(text)
        print("🤖: ",text)
        engine.runAndWait()
        
if __name__ == "__main__":
    speak("The feeling when knee surgery is tomorrow")