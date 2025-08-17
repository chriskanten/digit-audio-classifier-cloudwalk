#LLM Coding Challenge - Digit Classification from Audio

## 1. Project Structure & Setup
Create a project folder: e.g., digit-audio-classifier-using-llm/
Install dependencies to build the env.
```
pip install -r requirements.txt
```

## 2. Load FSDD dataset
Use ```datasets.load_dataset``` to load FSDD from Hugging Face.
You can easily cast the audio column to 8 kHz sampling rate with ```.cast_column()```.
In order to collect the dataset without downloading, you need to install and use the torchcodec.

### Note: 

1. The torchcodec currently requires GPU support for most of its functionality, especially for decoding audio or video data. On a CPU-only machine, you may encounter errors like:
```
ImportError: To support decoding audio data, please install 'torchcodec'.
```

2. Also, the torchcodec does not support Python 3.11. It was primarily built for Python 3.9 or 3.10, and using it in 3.11 will often give import errors like:
```
ImportError: cannot import name 'AudioDecoder' from 'torchcodec.decoders'
```
If you use the Google Colab, you might switch to the python3.10.
```
!python3.10 -m pip install torch torchvision torchaudio
!python3.10 -m pip install torchcodec
```

## 3. Preprocess audio: Mel Spectrogram features
We'll use librosa for MFCCs or Mel Spectrograms, which works well with torch models.

## 4. Build a simple CNN classifier
Build a lightweight CNN (two conv layers + fully connected head) tailored for Mel spectrogram input.

## 6. Train & evaluate
Standard training loop with loss tracking and test accuracy.

## 7. Live Digit Recognition
Use sounddevice to record audio.
Use same feature extraction as above.
Pass features to trained model for prediction.

# Summary Checklist
 Load FSDD dataset via Hugging Face
 Extract audio features (MFCC)
 Build simple MLP model in PyTorch
 Train on dataset
 Integrate live audio prediction via sounddevice
 Run everything on Windows, Python 3.11, torch 2.5+cuda12.1, without torchcodec

Here is the video to show total programming.
https://drive.google.com/file/d/1p2MhYpb86R0h-1XhjIJL0mHOJ-ZyUPpF/view?usp=sharing

#### By Chris
