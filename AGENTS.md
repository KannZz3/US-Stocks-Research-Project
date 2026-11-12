# Project conventions

- Use English for code, comments, documentation, output fields, and commit messages.
- Chinese may be used when communicating with the user in chat.
- Preserve `data.csv` as the unmodified source dataset.
- Rebuild processed data with `python prepare_data.py` after changing preparation logic.
- Run `python -m unittest discover -s tests -v` before committing preparation changes.
- Keep environments, credentials, and local caches out of Git.
