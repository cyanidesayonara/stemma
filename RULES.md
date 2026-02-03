# Project Rules & Conventions

## General Rules
1. **No Emojis:** Absolutely no emojis in code files, documentation, or commit messages. Keep it professional.
2. **Commit Frequently:** Commit after major changes or completed features to keep the Git log clean and PRs manageable.
3. **Conventional Commits:** Follow standard conventions for commit messages (e.g., `feat:`, `fix:`, `chore:`, `refactor:`, `docs:`).
4. **CLI Commands:** Execute only single CLI commands at a time during agentic interactions. Avoid chained commands (no `&&` or `;`).

## Code Style
- Follow PEP 8 guidelines for Python code.
- Provide clear docstrings for classes and complex functions.
- Avoid unnecessary dependencies; prefer a lean environment (e.g., ONNX Runtime over full PyTorch).
