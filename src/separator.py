import os
import numpy as np
import soundfile as sf
import onnxruntime as ort
from PySide6.QtCore import QThread, Signal

class SeparatorWorker(QThread):
    """
    Background thread for running ONNX inference to avoid freezing the UI.
    Emits progress updates and completion signals.
    """
    progress = Signal(int, str)  # percentage, status message
    finished = Signal(dict)      # mapping of stem_name -> file_path
    error = Signal(str)          # error message

    def __init__(self, input_path, output_dir, model_path, is_6_stem=False):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.model_path = model_path
        self.is_6_stem = is_6_stem
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            self.progress.emit(0, "Loading audio file...")
            # Load audio file
            audio, sr = sf.read(self.input_path, always_2d=True)
            audio = audio.T  # Shape: (channels, samples)
            
            # Resample if not 44100Hz (Demucs expects 44.1kHz typically)
            # TODO: implement resampling if sr != 44100
            
            self.progress.emit(10, "Initializing AI model (DirectML)...")
            # Setup ONNX Runtime session with DirectML
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            session_options = ort.SessionOptions()
            session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
            
            # The stem names mapping
            if self.is_6_stem:
                stems = ["vocals", "drums", "bass", "other", "guitar", "piano"]
            else:
                stems = ["vocals", "drums", "bass", "other"]

            # TODO: Add Chunking, STFT, Model Inference, and iSTFT here

            # Mock step for now
            self.progress.emit(50, "Separating stems...")
            import time
            time.sleep(2)  # Simulate processing
            
            if self._is_cancelled:
                self.error.emit("Separation cancelled by user.")
                return

            self.progress.emit(90, "Saving separated files...")
            os.makedirs(self.output_dir, exist_ok=True)
            result_files = {}
            
            # Mock savings
            for stem in stems:
                out_path = os.path.join(self.output_dir, f"{stem}.wav")
                # write silence for now
                sf.write(out_path, np.zeros((1000, 2)), 44100)
                result_files[stem] = out_path

            self.progress.emit(100, "Done")
            self.finished.emit(result_files)
            
        except Exception as e:
            self.error.emit(str(e))
