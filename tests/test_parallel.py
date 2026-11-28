import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import optuna

from parallel import Board, keep_file, sampler, search, shared, storage


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return (x - 1) ** 2


def search_worker(path, worker, workers, trials):
    db = storage(path)
    study = optuna.create_study(study_name="s", storage=db, load_if_exists=True, direction="minimize")
    study.sampler = sampler(study, worker)
    done = search(study, objective, trials, wait=lambda _: None)
    return [t.number for t in done], [t.params["x"] for t in study.trials]


def shared_worker(path, worker, workers, log):
    board = Board(path, "m")

    def run(job):
        with open(log, "a") as f:
            f.write(f"{job}\n")
        return job * 10

    return shared(board, "step", list(range(7)), run, worker, workers, wait=lambda _: None)


def in_processes(fn, workers, *args):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return [f.result() for f in [pool.submit(fn, *args[:1], w, workers, *args[1:]) for w in range(workers)]]


def test_workers_share_one_search_and_all_get_the_same_first_trials(tmp_path):
    results = in_processes(search_worker, 3, tmp_path / "study.db", 12)
    numbers = [r[0] for r in results]
    assert all(n == numbers[0] for n in numbers) and numbers[0] == sorted(numbers[0]) and len(numbers[0]) == 12
    xs = results[0][1]
    assert len(set(xs)) == len(xs)  # each worker draws its own settings, none repeated


def test_shared_jobs_run_once_each_and_every_worker_gets_every_result(tmp_path):
    log = tmp_path / "ran.txt"
    results = in_processes(shared_worker, 3, tmp_path / "board.db", log)
    assert all(r == [j * 10 for j in range(7)] for r in results)
    assert sorted(int(line) for line in log.read_text().split()) == list(range(7))
    # started again, the stored results are reused and nothing runs twice
    assert shared_worker(tmp_path / "board.db", 0, 3, log) == [j * 10 for j in range(7)]
    assert len(log.read_text().split()) == 7


def test_one_worker_runs_everything_here_and_stores_nothing():
    assert shared(None, "step", [1, 2], lambda job: job + 1) == [2, 3]


def test_one_worker_on_a_new_study_uses_the_seed_itself():
    study = optuna.create_study()
    assert sampler(study, 0)._rng.rng.randint(1_000_000) == optuna.samplers.TPESampler(seed=42)._rng.rng.randint(1_000_000)


def slow_worker(path):
    import time

    db = optuna.storages.RDBStorage(f"sqlite:///{path}", heartbeat_interval=1, grace_period=2)
    study = optuna.create_study(study_name="r", storage=db, load_if_exists=True)
    study.optimize(lambda trial: time.sleep(60) or trial.suggest_float("x", 0, 1), n_trials=1)


def test_a_trial_left_unfinished_by_a_stopped_worker_is_run_again_by_another(tmp_path):
    import time

    path = tmp_path / "study.db"
    process = multiprocessing.get_context("spawn").Process(target=slow_worker, args=(path,))
    process.start()
    db = optuna.storages.RDBStorage(f"sqlite:///{path}", heartbeat_interval=1, grace_period=2)
    while True:  # wait until the slow worker's trial has started, then stop that worker
        try:
            study = optuna.load_study(study_name="r", storage=db)
            if study.trials:
                break
        except KeyError:
            pass
        time.sleep(0.5)
    time.sleep(2)
    process.kill()
    done = search(study, objective, 3, wait=lambda _: time.sleep(1))
    states = [t.state for t in study.trials]
    assert len(done) == 3 and states[0] == optuna.trial.TrialState.FAIL and 0 not in [t.number for t in done]


def test_kept_files_come_back_from_the_board(tmp_path):
    board = Board(tmp_path / "board.db", "lightgbm_x")
    path = tmp_path / "tuned.json"
    path.write_text('{"a": 1}\n')
    keep_file(board, path)
    assert Board(tmp_path / "board.db", "anything").files() == {str(path): '{"a": 1}\n'}
