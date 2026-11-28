"""Run by pytest before any test is imported. The tests only ever use the local test database: settings that point
somewhere else, such as the shared research database a terminal is set up with for the model runs, are removed first,
so running the tests there can't read from or write to that database."""
import os

for name in ("DATABASE_URL", "OPTUNA_STORAGE", "SHARED_DATABASE_URL"):
    os.environ.pop(name, None)
