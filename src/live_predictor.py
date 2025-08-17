import sounddevice as sd
import numpy as np
import torch
from .feature_extractor import FeatureExtractor

class LivePredictor:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.feature_extractor = FeatureExtractor()
        self.device = device
        self.samplerate = 8000    

    def record_and_predict(self, duration=1):
        """ Record audio and predict the digit. """
        print("Recording...")
        audio_data = sd.rec(int(duration * self.samplerate), samplerate=self.samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")
        
        return audio_data.flatten()
    
    def predict(self, audio_data=None):
        """ Predict the digit from live audio data. """
        if audio_data is None:
            audio_data = self.record_and_predict()
        features = self.feature_extractor.extract_features(audio_data)        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.model.eval()        

        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_digit = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_digit].item()

        return predicted_digit, confidence
