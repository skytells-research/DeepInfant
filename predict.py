import torch
import librosa
import numpy as np
from pathlib import Path
from train import DeepInfantModel  # Import the model architecture

class InfantCryPredictor:
    def __init__(self, model_path='deepinfant.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize model
        self.model = DeepInfantModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (inverse of training mapping)
        self.label_map = {
            0: 'hungry',
            1: 'needs burping',
            2: 'belly pain',
            3: 'discomfort',
            4: 'tired'
        }
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            fmin=20,
            fmax=8000
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec)
    
    def predict(self, audio_path):
        """
        Predict the class of a single audio file
        Returns tuple of (predicted_label, confidence)
        """
        # Process audio
        mel_spec = self._process_audio(audio_path)
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        mel_spec = mel_spec.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][pred_class].item()
        
        return self.label_map[pred_class], confidence
    
    def predict_batch(self, audio_dir, file_extensions=('.wav', '.caf', '.3gp')):
        """
        Predict classes for all audio files in a directory
        Returns list of tuples (filename, predicted_label, confidence)
        """
        results = []
        audio_dir = Path(audio_dir)
        
        for audio_file in audio_dir.glob('*.*'):
            if audio_file.suffix.lower() in file_extensions:
                label, confidence = self.predict(str(audio_file))
                results.append((audio_file.name, label, confidence))
        
        return results

def main():
    # Example usage
    predictor = InfantCryPredictor()
    
    # Single file prediction
    audio_path = "path/to/your/audio.wav"
    label, confidence = predictor.predict(audio_path)
    print(f"\nPrediction for {audio_path}:")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2%}")
    
    # Batch prediction
    audio_dir = "path/to/audio/directory"
    results = predictor.predict_batch(audio_dir)
    print("\nBatch Predictions:")
    for filename, label, confidence in results:
        print(f"{filename}: {label} ({confidence:.2%})")

if __name__ == "__main__":
    main() 