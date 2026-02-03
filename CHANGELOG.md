# Changelog

All notable development sessions are documented here in reverse chronological order.

---

## 2026-03-20 -- Session 1: Project Setup

### Done
- Researched stem separation technology landscape (HTDemucs v4, ONNX Runtime, existing projects)
- Chose tech stack: Python 3.14, PySide6, ONNX Runtime + DirectML, sounddevice
- Created `PROJECT.md` with full spec, module descriptions, and phased roadmap
- Created GitHub repo (https://github.com/cyanidesayonara/stemma), pushed initial scaffold
- Set up project folder structure with all source file skeletons
- Created virtual environment and installed dependencies
- Started `separator.py` (SeparatorWorker QThread skeleton) and `model_manager.py` (ModelDownloader)
- Adopted `AGENTS.md` open standard for cross-tool AI handover
- Set up GitHub Projects kanban board with Phase 1 issues

### Next
- Complete ONNX inference pipeline in `separator.py`
- Implement multi-track audio player (`player.py`)
- Build UI components
