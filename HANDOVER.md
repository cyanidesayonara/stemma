# Welcome, Future AI Assistant!

You are picking up the **stemma** project, a local-only Windows desktop music player with AI stem separation.

To get fully up to speed, please read the documentation in this exact order:

1. **`PROJECT.md`** — Contains the core concept, architecture, module specifications, and the Phase 1/2/3 roadmap.
2. **`RULES.md`** — Contains the strict project rules (NO EMOJIS, single CLI commands only, conventional commits). You must follow these closely.
3. **`task.md` (or your internal artifact equivalent)** — Check this to see what exact sub-tasks have been completed.

### Current Status (as of March 20, 2026)

- **Completed:** Repository created, basic folder structure scaffolded (`src/`, `data/`, etc.), virtual environment checked in `.venv`, and `requirements.txt` written. `PROJECT.md` and `RULES.md` are finalized.
- **In Progress:** `Phase 1 -> Implement stem separation engine (separator.py)`. The placeholder class `SeparatorWorker` exists in `src/separator.py` and `ModelManager` exists in `src/model_manager.py`.
- **Next Steps:** Complete the Demucs ONNX inference logic using `onnxruntime` and `librosa` (for STFT) in `separator.py`. 

Please continue from the "Next Steps". Check `PROJECT.md` for specific technical decisions regarding `separator.py`.
