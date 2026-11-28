"""Run a model's work on several machines (or processes) at once, all sharing one database, so every model's search runs
the same way: 4 trials at a time. Every worker runs the same command with its own --worker number (0 to --workers - 1).
Finished trials and results are stored in the shared database, and each step waits until every worker's part is done,
so all the workers carry on from the same results. With one worker, everything runs in this process.

The database is OPTUNA_STORAGE (a Postgres URL every machine can reach), or a local SQLite file when it isn't set (for
several processes on one machine). A worker that stops can be started again: finished trials and stored results are
kept, and a trial left unfinished by a stopped worker is marked failed after a few minutes and run again by another."""
import json
import os
import time
import warnings

import optuna

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # heartbeats are marked experimental
poll_seconds = 30
counted = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING)


def database_url(local_path):
    """The shared database: OPTUNA_STORAGE if it's set, otherwise a SQLite file at local_path."""
    return os.environ.get("OPTUNA_STORAGE") or f"sqlite:///{local_path}"


def retried(make, attempts=10):
    """Workers starting together can race to create the same tables: wait and retry."""
    for attempt in range(attempts):
        try:
            return make()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)


def storage(local_path):
    """The searches' database. Each running trial reports every minute; one silent for 5 minutes (its worker stopped) is
    marked failed so another worker reruns it."""
    return retried(lambda: optuna.storages.RDBStorage(database_url(local_path), heartbeat_interval=60, grace_period=300,
                                                      engine_kwargs={"pool_pre_ping": True}))


class Board:
    """Results shared between the workers: a table of keys and JSON text in the shared database, under a name for the
    model, so a waiting worker only checks which results are ready and reads each once."""

    def __init__(self, local_path, name):
        from sqlalchemy import create_engine, text

        self.engine, self.name, self.text = create_engine(database_url(local_path), pool_pre_ping=True), name, text

        def create():
            with self.engine.begin() as connection:
                connection.execute(text("CREATE TABLE IF NOT EXISTS shared_results (key VARCHAR(400) PRIMARY KEY, "
                                        "value TEXT NOT NULL)"))
        retried(create)

    def key(self, tag, job):
        return f"{self.name}|{tag}|{json.dumps(job)}"

    def put(self, key, value):
        with self.engine.begin() as connection:
            connection.execute(self.text("INSERT INTO shared_results (key, value) VALUES (:key, :value) ON CONFLICT (key) "
                                         "DO UPDATE SET value = excluded.value"), {"key": key, "value": json.dumps(value)})

    def ready(self, keys):
        """Which of keys have a stored result."""
        from sqlalchemy import bindparam

        query = self.text("SELECT key FROM shared_results WHERE key IN :keys").bindparams(bindparam("keys", expanding=True))
        with self.engine.connect() as connection:
            return {row[0] for row in connection.execute(query, {"keys": list(keys)})}

    def get(self, keys):
        from sqlalchemy import bindparam

        query = self.text("SELECT key, value FROM shared_results WHERE key IN :keys").bindparams(
            bindparam("keys", expanding=True))
        with self.engine.connect() as connection:
            return {row[0]: json.loads(row[1]) for row in connection.execute(query, {"keys": list(keys)})}

    def files(self):
        """Every file kept with keep_file, as {path: text}, whatever the model."""
        with self.engine.connect() as connection:
            rows = connection.execute(self.text("SELECT key, value FROM shared_results WHERE key LIKE '%|file|%'"))
            return {row[0].split("|file|", 1)[1]: json.loads(row[1]) for row in rows}


def sampler(study, worker, seed=42):
    """Each worker draws its own settings: its own seed, moved on by the trials already in the study, so a restarted
    search doesn't repeat the settings it tried. One worker on a new study gets exactly `seed`."""
    return optuna.samplers.TPESampler(seed=seed + 1000 * worker + len(study.trials))


def search(study, objective, trials, wait=time.sleep):
    """Run trials until `trials` have been started across all the workers, wait for the other workers to finish theirs,
    and return the first `trials` finished trials (by trial number): the same trials on every worker. A trial that
    failed (its worker stopped) doesn't count, so another is run in its place."""
    while True:
        while len(study.get_trials(deepcopy=False, states=counted)) < trials:
            study.optimize(objective, n_trials=1)
        while study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,)):
            optuna.storages.fail_stale_trials(study)
            wait(poll_seconds)
        done = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if len(done) >= trials:
            return sorted(done, key=lambda t: t.number)[:trials]


def shared(board, tag, jobs, run, worker=0, workers=1, wait=time.sleep):
    """Run `run(job)` for every job, split across the workers (job i on worker i % workers), and return every job's
    result in job order, on every worker, once all are done. Jobs and results must be JSON (lists, numbers, strings,
    dicts). Results are kept on `board` under `tag`, so a worker started again only runs the jobs not yet stored. With
    one worker, everything runs here and nothing is stored."""
    if workers == 1:
        return [run(job) for job in jobs]
    keys = [board.key(tag, job) for job in jobs]
    stored = board.ready(keys)
    for i, (job, key) in enumerate(zip(jobs, keys)):
        if i % workers == worker and key not in stored:
            board.put(key, run(job))
    while len(board.ready(keys)) < len(keys):
        wait(poll_seconds)
    results = board.get(keys)
    return [results[key] for key in keys]


def keep_file(board, path):
    """Store a saved file's text on the board, so it can be brought back to another machine (src/copy_to_shared.py
    --back) without copying it from the worker that wrote it."""
    board.put(f"{board.name}|file|{path}", path.read_text())
