import librosa
import numpy as np
import argparse
import os
#import data_form
import pickle

"""
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
def extract_feature(file_name, **kwargs):
    
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X)) #analyze the frequency content of a signal over time
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result

#parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
#                                    and perform inference on a sample you provide (either using your voice or a file)""")
#parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
#args = parser.parse_args()
#file = args.file
# construct the model
#model = data_form.create_model()
# load the saved/trained weights
#model.load_weights("results/model.h5")

def test_sample(file):
    if not file or not os.path.isfile(file):
        return "file doesn't exist"
    else:
        loaded_model = pickle.load(open("gender.pickle", 'rb'))
        features = extract_feature(file, mel=True).reshape(1, -1)
        # predict the gender!
        print(loaded_model.predict(features))
        male_prob = loaded_model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        # show the result!
        #print("Result:", gender)
        #print(f"Probabilities::: Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")
        result = "Result: " + gender + ", Probabilities::: Male: " + f"{male_prob*100:.2f}" + "   Female: " + f"{female_prob*100:.2f}"
        print(result)
        return result