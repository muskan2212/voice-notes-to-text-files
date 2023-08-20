import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sr
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer


file_name = 'voice.wav'
Audio(file_name)

r = sr.Recognizer()

with sr.AudioFile(file_name) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    print(text)
    
text_set=set()

for t in text:
    text_set.add(t)
    
print("total different words:",len(text_set))