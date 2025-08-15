"""
Voice Text Processor Module

A comprehensive module for converting voice input to text with support for
Hinglish, Hindi, and English languages. Includes noise handling, language
detection, and punctuation restoration.

Dependencies:
    pip install transformers torch torchaudio librosa soundfile numpy scipy
    pip install langdetect punctuator
"""

import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    pipeline
)
from scipy.signal import butter, filtfilt
from langdetect import detect
import re
import warnings
warnings.filterwarnings("ignore")


class VoiceTextProcessor:
    """
    A multilingual voice-to-text processor supporting Hindi, English, and Hinglish.
    """
    
    def __init__(self, model_name="openai/whisper-small", device=None):
        """
        Initialize the voice text processor.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
            device (str): Device to run the model on ('cpu', 'cuda', or None for auto)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize primary model (Whisper for multilingual support)
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.asr_pipeline = None
        
        # Language detection patterns
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the specified model and processor."""
        try:
            if "whisper" in self.model_name.lower():
                print(f"Loading Whisper model: {self.model_name}")
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.to(self.device)
                
                # Create ASR pipeline for easier inference
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    return_timestamps=True
                )
            
            elif "wav2vec2" in self.model_name.lower():
                print(f"Loading Wav2Vec2 model: {self.model_name}")
                self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
                self.model.to(self.device)
            
            else:
                # Fallback to whisper-small
                print("Unknown model, falling back to whisper-small")
                self.model_name = "openai/whisper-small"
                self._load_model()
                
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            print("Falling back to whisper-small")
            self.model_name = "openai/whisper-small"
            self._load_model()
    
    def preprocess_audio(self, audio_path_or_array, target_sr=16000):
        """
        Preprocess audio file or array with noise reduction and normalization.
        
        Args:
            audio_path_or_array: Path to audio file or numpy array
            target_sr (int): Target sample rate
            
        Returns:
            tuple: (processed_audio_array, sample_rate)
        """
        # Load audio
        if isinstance(audio_path_or_array, str):
            audio, sr = librosa.load(audio_path_or_array, sr=target_sr)
        else:
            audio = audio_path_or_array
            sr = target_sr
        
        # Ensure mono audio
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Apply noise reduction (simple high-pass filter)
        audio = self._apply_noise_reduction(audio, sr)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        return audio, sr
    
    def _apply_noise_reduction(self, audio, sr, cutoff_freq=100):
        """
        Apply basic noise reduction using high-pass filter.
        
        Args:
            audio: Audio array
            sr: Sample rate
            cutoff_freq: High-pass filter cutoff frequency
            
        Returns:
            numpy.ndarray: Filtered audio
        """
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Design high-pass filter
        b, a = butter(N=5, Wn=normalized_cutoff, btype='high')
        
        # Apply filter
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def detect_language(self, text):
        """
        Detect the primary language of the transcribed text.
        
        Args:
            text (str): Transcribed text
            
        Returns:
            str: Detected language ('hindi', 'english', or 'hinglish')
        """
        if not text.strip():
            return 'unknown'
        
        # Check for Hindi characters
        hindi_chars = len(self.hindi_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        # Simple heuristic for language detection
        if hindi_chars > 0 and english_chars > 0:
            return 'hinglish'
        elif hindi_chars > english_chars:
            return 'hindi'
        elif english_chars > 0:
            return 'english'
        else:
            try:
                detected = detect(text)
                if detected == 'hi':
                    return 'hindi'
                elif detected == 'en':
                    return 'english'
                else:
                    return 'unknown'
            except:
                return 'unknown'
    
    def restore_punctuation(self, text, language='english'):
        """
        Basic punctuation restoration using simple rules.
        
        Args:
            text (str): Text without punctuation
            language (str): Language of the text
            
        Returns:
            str: Text with restored punctuation
        """
        if not text:
            return text
        
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
        
        # Add period at the end if missing
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Simple rules for common patterns
        text = re.sub(r'\bहैं\b(?!\s*[।.])', 'हैं।', text)  # Hindi sentence ending
        text = re.sub(r'\bहै\b(?!\s*[।.])', 'है।', text)    # Hindi sentence ending
        text = re.sub(r'\byes\b', 'Yes', text, flags=re.IGNORECASE)
        text = re.sub(r'\bno\b', 'No', text, flags=re.IGNORECASE)
        
        return text
    
    def transcribe_with_whisper(self, audio_array, sr=16000):
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_array: Preprocessed audio array
            sr: Sample rate
            
        Returns:
            dict: Transcription results with text, language, and confidence
        """
        try:
            # Use the ASR pipeline for easier inference
            result = self.asr_pipeline(
                audio_array,
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps=True
            )
            
            text = result['text']
            
            # Detect language from the transcribed text
            detected_language = self.detect_language(text)
            
            return {
                'text': text.strip(),
                'language': detected_language,
                'confidence': 0.8,  # Whisper doesn't provide confidence scores directly
                'chunks': result.get('chunks', [])
            }
            
        except Exception as e:
            print(f"Error in Whisper transcription: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'chunks': []
            }
    
    def transcribe_with_wav2vec2(self, audio_array, sr=16000):
        """
        Transcribe audio using Wav2Vec2 model.
        
        Args:
            audio_array: Preprocessed audio array
            sr: Sample rate
            
        Returns:
            dict: Transcription results
        """
        try:
            # Process audio
            inputs = self.processor(audio_array, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Decode predictions
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Detect language
            detected_language = self.detect_language(transcription)
            
            return {
                'text': transcription.strip(),
                'language': detected_language,
                'confidence': 0.7,  # Approximate confidence
                'chunks': []
            }
            
        except Exception as e:
            print(f"Error in Wav2Vec2 transcription: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'chunks': []
            }
    
    def transcribe(self, audio_input, add_punctuation=True):
        """
        Main transcription method with automatic language detection.
        
        Args:
            audio_input: Audio file path or numpy array
            add_punctuation (bool): Whether to restore punctuation
            
        Returns:
            dict: Complete transcription results
        """
        try:
            # Preprocess audio
            audio_array, sr = self.preprocess_audio(audio_input)
            
            # Choose transcription method based on model type
            if "whisper" in self.model_name.lower():
                result = self.transcribe_with_whisper(audio_array, sr)
            else:
                result = self.transcribe_with_wav2vec2(audio_array, sr)
            
            # Restore punctuation if requested
            if add_punctuation and result['text']:
                result['text'] = self.restore_punctuation(result['text'], result['language'])
            
            # Add metadata
            result.update({
                'model_used': self.model_name,
                'sample_rate': sr,
                'duration': len(audio_array) / sr,
                'preprocessing_applied': True
            })
            
            return result
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return {
                'text': '',
                'language': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def transcribe_file(self, file_path, add_punctuation=True):
        """
        Transcribe an audio file.
        
        Args:
            file_path (str): Path to the audio file
            add_punctuation (bool): Whether to restore punctuation
            
        Returns:
            dict: Transcription results
        """
        return self.transcribe(file_path, add_punctuation)
    
    def transcribe_array(self, audio_array, sample_rate=16000, add_punctuation=True):
        """
        Transcribe an audio array.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate (int): Sample rate of the audio
            add_punctuation (bool): Whether to restore punctuation
            
        Returns:
            dict: Transcription results
        """
        return self.transcribe(audio_array, add_punctuation)


# Utility functions for easy usage
def create_processor(model_name="openai/whisper-small", device=None):
    """
    Create a VoiceTextProcessor instance with specified model.
    
    Args:
        model_name (str): Hugging Face model name
        device (str): Device to use
        
    Returns:
        VoiceTextProcessor: Configured processor instance
    """
    return VoiceTextProcessor(model_name, device)


def quick_transcribe(audio_input, model_name="openai/whisper-small"):
    """
    Quick transcription function for one-off usage.
    
    Args:
        audio_input: Audio file path or numpy array
        model_name (str): Model to use
        
    Returns:
        str: Transcribed text
    """
    processor = create_processor(model_name)
    result = processor.transcribe(audio_input)
    return result.get('text', '')


# Example usage
if __name__ == "__main__":
    # Example 1: Using Whisper model
    print("Creating voice text processor with Whisper...")
    processor = VoiceTextProcessor("openai/whisper-small")
    
    # Example 2: Transcribe a file (replace with actual audio file path)
    # result = processor.transcribe_file("example_audio.wav")
    # print(f"Transcription: {result['text']}")
    # print(f"Language: {result['language']}")
    # print(f"Confidence: {result['confidence']}")
    
    # Example 3: Quick transcribe
    # text = quick_transcribe("example_audio.wav")
    # print(f"Quick transcription: {text}")
    
    # Example 4: Using different models
    available_models = [
        "openai/whisper-small",
        "openai/whisper-base",
        "facebook/wav2vec2-large-xlsr-53-hindi",
        "facebook/wav2vec2-large-960h-lv60-self"  # English
    ]
    
    print("Available models for different use cases:")
    for model in available_models:
        print(f"- {model}")
    
    print("\nProcessor created successfully!")
    print("Use processor.transcribe_file('your_audio.wav') to transcribe audio files.")