import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf

def prepare_dataset(source_dir="Data/v2", output_dir="processed_dataset", test_size=0.2):
    """
    Prepare the dataset by organizing and splitting audio files.
    
    Args:
        source_dir (str): Source directory containing class folders
        output_dir (str): Output directory for processed dataset
        test_size (float): Proportion of data to use for testing
    """
    # Create output directories
    output_path = Path(output_dir)
    train_path = output_path / "train"
    test_path = output_path / "test"
    
    for path in [train_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
        
    # Define class mapping
    class_mapping = {
        'belly_pain': 'bp',
        'burping': 'bu',
        'cold_hot': 'ch',
        'discomfort': 'dc',
        'hungry': 'hu',
        'lonely': 'lo',
        'scared': 'sc',
        'tired': 'ti',
        'unknown': 'un'
    }
    
    # Create metadata list
    metadata = []
    
    # Process each class folder
    source_path = Path(source_dir)
    for class_folder in source_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            class_code = class_mapping[class_name]
            
            # Get all audio files in the class folder
            audio_files = list(class_folder.glob("*.wav")) + \
                         list(class_folder.glob("*.mp3")) + \
                         list(class_folder.glob("*.caf")) + \
                         list(class_folder.glob("*.3gp"))
            
            # Split files into train and test
            train_files, test_files = train_test_split(
                audio_files, test_size=test_size, random_state=42
            )
            
            # Process and copy files
            for files, split_path in [(train_files, train_path), (test_files, test_path)]:
                for audio_file in files:
                    # Load and resample audio to 16kHz
                    try:
                        y, sr = librosa.load(audio_file, sr=16000)
                        
                        # Generate new filename
                        new_filename = f"{audio_file.stem}-{class_code}.wav"
                        output_file = split_path / new_filename
                        
                        # Save processed audio
                        sf.write(output_file, y, sr, subtype='PCM_16')
                        
                        # Add to metadata
                        metadata.append({
                            'filename': new_filename,
                            'class': class_name,
                            'class_code': class_code,
                            'split': 'train' if split_path == train_path else 'test'
                        })
                        
                    except Exception as e:
                        print(f"Error processing {audio_file}: {str(e)}")
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path / 'metadata.csv', index=False)
    
    print(f"Dataset prepared successfully in {output_dir}")
    print("\nClass distribution:")
    print(metadata_df.groupby(['split', 'class']).size().unstack(fill_value=0))

if __name__ == "__main__":
    prepare_dataset() 