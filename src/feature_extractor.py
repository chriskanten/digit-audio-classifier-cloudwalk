import librosa
import numpy as np

class FeatureExtractor:
    def __init__(self, n_mfcc=13, sr=8000):
        self.n_mfcc = n_mfcc
        self.sr = sr

    def extract_features(self, audio_path):
        """ Extract MFCC features from an audio file."""
        
        mfcc = librosa.feature.mfcc(
            y=audio_path, 
            sr=self.sr, n_mfcc=self.n_mfcc
        )

        return np.mean(mfcc.T, axis=0)  
    
    def process_audio(self, batch):
        """ Process the audio file and extract features."""
        
        feactures = []
        for audio_path in batch:
            mfcc_features = self.extract_features(audio_path)
            feactures.append(mfcc_features)
        
        return np.array(feactures)  # shape: (batch_size, n_mfcc)
