import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DeepInfantDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Updated label mapping based on new classes
        self.label_map = {
            'bp': 0,  # belly pain
            'bu': 1,  # burping
            'ch': 2,  # cold/hot
            'dc': 3,  # discomfort
            'hu': 4,  # hungry
            'lo': 5,  # lonely
            'sc': 6,  # scared
            'ti': 7,  # tired
            'un': 8,  # unknown
        }
        
        # Load metadata if available
        metadata_file = Path(data_dir).parent / 'metadata.csv'
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row['split'] == self.data_dir.name:  # 'train' or 'test'
                audio_path = self.data_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    self.labels.append(self.label_map[row['class_code']])
    
    def _load_dataset(self):
        for audio_file in self.data_dir.glob('*.*'):
            if audio_file.suffix in ['.wav', '.caf', '.3gp']:
                # Parse filename for label
                label = audio_file.stem.split('-')[-1][:2]  # Get reason code
                if label in self.label_map:
                    self.samples.append(str(audio_file))
                    self.labels.append(self.label_map[label])
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Add basic audio augmentation (during training)
        if self.transform:
            # Random time shift (-100ms to 100ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[:len(waveform)]
            else:
                waveform = np.pad(waveform, (0, -shift))[(-shift):]
            
            # Random noise injection
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(waveform))
                waveform = waveform + noise
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram with adjusted parameters
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,  # Reduced from 2048 for better temporal resolution
            hop_length=256,  # Reduced from 512
            n_mels=80,  # Standard for speech/audio
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency, suitable for infant cries
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]
        
        # Process audio to mel spectrogram
        mel_spec = self._process_audio(audio_path)
        
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, label

class DeepInfantModel(nn.Module):
    def __init__(self, num_classes=9):
        super(DeepInfantModel, self).__init__()
        
        # CNN layers with residual connections
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Adding squeeze-and-excitation block
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 256, 1),
            nn.Sigmoid()
        )
        
        # Bi-directional LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=256 * 10,  # Adjusted based on new mel spec parameters
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),  # 1024 due to bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch, 1, freq_bins, time_steps)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1, 3)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, -1, 256 * 10)  # (batch, time, features)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step
        
        # Classification
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'deepinfant.pth')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets using processed data
    train_dataset = DeepInfantDataset('processed_dataset/train', transform=True)
    val_dataset = DeepInfantDataset('processed_dataset/test', transform=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model, loss function, and optimizer
    model = DeepInfantModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

if __name__ == '__main__':
    main() 