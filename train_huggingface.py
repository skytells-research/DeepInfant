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
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path("./deepinfant/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset from HuggingFace
    dataset = load_dataset("your_username/your_dataset")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Split dataset
    train_dataset = DeepInfantHFDataset(dataset['train'])
    eval_dataset = DeepInfantHFDataset(dataset['validation'])
    
    # Configure model with custom config instead of BERT
    config = AutoConfig.from_dict({
        "num_labels": 5,  # Number of cry classifications
        "hidden_size": 512,  # Match LSTM hidden size
        "num_attention_heads": 8,
        "num_hidden_layers": 2,  # Match LSTM layers
        "model_type": "deepinfant",
        "architectures": ["DeepInfantHFModel"]
    })
    model = DeepInfantHFModel(config)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./deepinfant",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=True,
        hub_model_id="deepinfant",
        logging_dir='./logs',
        logging_steps=50,
        resume_from_checkpoint=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Check for existing checkpoints
    last_checkpoint = None
    if checkpoint_dir.exists():
        checkpoints = [str(x) for x in checkpoint_dir.glob("checkpoint-*")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
            print(f"Resuming from checkpoint: {last_checkpoint}")
    
    # Train model
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Save final model
    trainer.save_model("./deepinfant/final")
    
    # Push model to hub
    trainer.push_to_hub()
    
    # Save additional checkpoint information
    checkpoint_info = {
        "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        "total_steps": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
    }
    
    with open(checkpoint_dir / "checkpoint_info.txt", "w") as f:
        for key, value in checkpoint_info.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    main() 