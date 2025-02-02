import os
from datasets import load_dataset, Audio
from transformers import Trainer, TrainingArguments
import torch
from train import DeepInfantModel, DeepInfantDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from pathlib import Path
import librosa
from torch.utils.data import Dataset
from transformers import PreTrainedModel, AutoConfig

class DeepInfantHFModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = DeepInfantModel(num_classes=config.num_labels)
        
    def forward(self, input_values, labels=None):
        outputs = self.model(input_values)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(outputs, labels)
            
        return {"loss": loss, "logits": outputs} if loss is not None else outputs

class DeepInfantHFDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def _process_audio(self, waveform, sample_rate):
        # Resample if necessary
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
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
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio = item['audio']
        waveform = audio['array']
        sample_rate = audio['sampling_rate']
        
        # Process audio to mel spectrogram
        mel_spec = self._process_audio(waveform, sample_rate)
        mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension
        
        return {
            "input_values": mel_spec,
            "labels": item['label']
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load dataset from HuggingFace
    dataset = load_dataset("your_username/your_dataset")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split dataset
    train_dataset = DeepInfantHFDataset(dataset['train'])
    eval_dataset = DeepInfantHFDataset(dataset['validation'])
    
    # Configure model
    config = AutoConfig.from_pretrained('bert-base-uncased')  # Using as base config
    config.num_labels = 5  # Number of cry classifications
    model = DeepInfantHFModel(config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./deepinfant",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
        logging_dir='./logs',
        hub_model_id="deepinfant",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    trainer.train()
    
    # Push model to hub
    trainer.push_to_hub()

if __name__ == "__main__":
    main() 