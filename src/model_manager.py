import os
import urllib.request
from PySide6.QtCore import QObject, Signal, QThread

class ModelDownloader(QThread):
    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, model_name, models_dir):
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        self.repo_url = "https://huggingface.co/rysertio/Demucs-onnx/resolve/main"

    def run(self):
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            file_name = f"{self.model_name}.onnx"
            url = f"{self.repo_url}/{file_name}"
            dest_path = os.path.join(self.models_dir, file_name)

            if os.path.exists(dest_path):
                self.progress.emit(100, f"{file_name} already exists.")
                self.finished.emit(dest_path)
                return

            self.progress.emit(0, f"Downloading {file_name}...")
            
            def report_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = int(block_num * block_size * 100 / total_size)
                    percent = min(100, percent)
                    self.progress.emit(percent, f"Downloading {file_name}... {percent}%")

            urllib.request.urlretrieve(url, dest_path, reporthook=report_hook)
            
            self.progress.emit(100, "Download complete.")
            self.finished.emit(dest_path)
            
        except Exception as e:
            self.error.emit(str(e))

class ModelManager(QObject):
    """
    Manages checking and downloading ONNX models.
    """
    def __init__(self, data_dir="data"):
        super().__init__()
        self.models_dir = os.path.join(data_dir, "models")
        self.downloader = None

    def is_model_downloaded(self, is_6_stem=False):
        model_name = "htdemucs_6s" if is_6_stem else "htdemucs"
        path = os.path.join(self.models_dir, f"{model_name}.onnx")
        return os.path.exists(path)

    def download_model(self, is_6_stem=False):
        model_name = "htdemucs_6s" if is_6_stem else "htdemucs"
        self.downloader = ModelDownloader(model_name, self.models_dir)
        return self.downloader
