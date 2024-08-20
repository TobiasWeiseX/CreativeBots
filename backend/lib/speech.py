import os, time
import os.path
from datetime import datetime, date
import pyttsx3


"""
flite -voice slt -t "This example is useful when there is a need to convert the contents of a file to speech. It can simplify tasks such as reading out the contents of a document or generating voiceovers for specific text files."
"""

#espeak -v mb-en1 -s 120 "Hello world"


def text_to_speech(text: str) -> str:
    unix_timestamp = datetime.now().timestamp()

    txt_file_path = "./public/speech_%s.txt" % unix_timestamp


    with open(txt_file_path, "w") as f:
        f.write(text)


    file_name = f'speech_{unix_timestamp}.wav'
    file_path = f'./public/{file_name}'
    #os.system('flite -voice slt -o %s -t "%s"' % (file_path, text))

    os.system('flite -voice slt -o %s -f %s' % (file_path, txt_file_path))

    return file_name



def text_to_speech2(text: str) -> str:
    engine = pyttsx3.init()

    def get_voice(s):
        for v in engine.getProperty("voices"):
            if s == v.id:
                return v

    def set_voice(v):
        engine.setProperty("voice", v.id)

    def set_volume(n):
        engine.setProperty('volume', engine.getProperty('volume') + n)

    def set_rate(n):
        engine.setProperty('rate', engine.getProperty('rate') + n)

    #voices = engine.getProperty('voices')
    #engine.setProperty('voice', voices[1].id)
    set_voice(get_voice("english"))
    set_volume(-5.0)
    set_rate(-40)

    #espeak -v mb-en1 -s 120 "Hello world"
    #sudo apt-get install mbrola mbrola-en1

    unix_timestamp = datetime.now().timestamp()
    file_name = f'speech_{unix_timestamp}.mp3'
    file_path = f'./public/{file_name}'

    engine.save_to_file(text, file_path)
    engine.runAndWait()

    timeout = 10
    t = 0
    step = 0.1
    while not os.path.isfile(file_path):
        time.sleep(step)
        t += step
        if t > timeout:
            raise Exception("Timeout(%s s) for creating speech.mp3!" % timeout)

    time.sleep(step)
    return file_name













