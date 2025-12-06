import datetime
import random
import webbrowser
import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
import tempfile, os

SRATE = 16000     # tasa de muestreo
DUR = 5           # segundos

print("Grabando... habla ahora!")
audio = sd.rec(int(DUR*SRATE), samplerate=SRATE, channels=1, dtype='int16')
sd.wait()
print("Listo, procesando...")

# guarda a WAV temporal
tmp_wav = tempfile.mktemp(suffix=".wav")
write(tmp_wav, SRATE, audio)

# reconoce con SpeechRecognition
r = sr.Recognizer()
with sr.AudioFile(tmp_wav) as source:
    data = r.record(source)

try:
    texto = r.recognize_google(data, language="es-ES")
    print("Dijiste:", texto)
except sr.UnknownValueError:
    print("No se entendió el audio.")
except sr.RequestError as e:
    print("Error:", e)
finally:
    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

# Procesar solo si hay texto reconocido
if texto:
    cmd = texto.lower()

    if "hola" in cmd:
        print("¡Hola, bienvenido al curso!")

    elif "abrir google" in cmd:
        webbrowser.open("https://www.google.com")

    elif "abrir youtube" in cmd:
        cancion = cmd.replace("abrir youtube", "").strip()
        if cancion:
            print(f"Buscando en YouTube: {cancion}")
            webbrowser.open(f"https://www.youtube.com/results?search_query={cancion}")
        else:
            webbrowser.open("https://www.youtube.com")

    elif "hora" in cmd:
        print("Hora actual:", datetime.now().strftime("%H:%M"))

    elif "contar un chiste" in cmd:
        chistes = [
            "¿Qué hace una abeja en el gimnasio? ¡Zum-ba!",
            "¿Cómo se despiden los químicos? Ácido un placer.",
            "¿Por qué estaba feliz la escoba? Porque ba-rriendo."
        ]
        print(random.choice(chistes))

    elif "adiós" in cmd or "salir" in cmd:
        print("¡Hasta luego! Fue un gusto hablar contigo.")
        exit()

    else:
        print("Comando no reconocido.")
