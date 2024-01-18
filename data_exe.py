import pandas as pd
import numpy as np
import os
import tqdm
import pickle

df = pd.read_csv("balanced-all.csv")
print(df.head())

label2int = {
    "male": 1,
    "female": 0
}

def load_data(vector_length=128):
    # get total samples
    n_samples = len(df)
    # get total male samples
    n_male_samples = len(df[df['gender'] == 'male'])
    # get total female samples
    n_female_samples = len(df[df['gender'] == 'female'])
    print("Total samples:", n_samples)
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    # initialize an empty array for all audio features
    X = np.zeros((n_samples, vector_length))
    # initialize an empty array for all audio labels (1 for male and 0 for female)
    y = np.zeros((n_samples, 1))
    TRAIN_PATH = '/home/clustox/Downloads/archive (2)/cv-valid-train/'
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(TRAIN_PATH + df['filename'], df['gender'])), "Loading data", total=n_samples):
        print(filename)
        features = np.load(filename, allow_pickle = True)
        X[i] = features
        y[i] = label2int[gender]
        print('11111111111')
    # save the audio features and labels into files
    # so we won't load each one of them next run
    np.save("results/features", X)
    np.save("results/labels", y)
    return X, y

X, y = load_data()
print(X)
print(y)