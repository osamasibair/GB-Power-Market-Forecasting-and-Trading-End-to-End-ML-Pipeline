# GB Power Market Forecasting and Trading: End-to-End ML Pipeline

[![CI](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/tests.yml)
[![Deploy](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/deploy.yml/badge.svg)](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/deploy.yml)
[![Daily forecast](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/daily.yml/badge.svg?event=schedule)](https://github.com/osamasibair/GB-Power-Market-Forecasting-and-Trading-End-to-End-ML-Pipeline/actions/workflows/daily.yml)

An end-to-end machine learning system that forecasts Great Britain’s electricity demand, prices and backtests battery trading, taken to production, then used as a testbed to compare gradient boosting with deep learning.

**Live:** [API docs](https://electricity-api-xqiuvbbyca-nw.a.run.app/docs) · [Latest forecast](https://electricity-api-xqiuvbbyca-nw.a.run.app/forecast/latest) · [Dashboard](https://electricity-dashboard-xqiuvbbyca-nw.a.run.app) (demand forecasting, live and on the test year, plus price, spike and battery results on the test year)

- **Demand: LightGBM day ahead model: 5.08% MAPE, about 39% lower error than a same-time-last-week baseline (8.39%), with 80% prediction intervals achieving 80.2% coverage.**
- **Price: LightGBM model with £21.50/MWh MAE, 15% lower than the best naive baseline, using wind, solar and net demand to forecast cheap and expensive periods.**
- **Spikes: LightGBM classifier that ranks spike risk 2.5x better than persistence (average precision 0.43 vs 0.17, ROC AUC 0.90).**
- **Backtest: a simulated 1 MW / 2 MWh battery scheduled from the price forecast earns 77% of the perfect foresight profit, against 52% for a baseline forecast and 74% for a simple average of the last 7 days prices.**
- **Deep learning: fine tuned Chronos 2, Amazon's time series foundation model fine tuned on a cloud GPU, is the best single model at 4.49% MAPE against 5.08% for LightGBM. A blend of LightGBM, a PatchTST transformer implemented from scratch in PyTorch, and fine tuned Chronos 2, with every choice made on a validation year, reaches 4.36%, 14% lower error than LightGBM alone.**

**Stack:** Python, pandas, LightGBM, PyTorch, Hugging Face (Chronos 2), Optuna, Google Colab, FastAPI, Docker, pytest, SciPy, Ruff, GitHub Actions, PostgreSQL, SQLAlchemy, Docker Compose, MLFlow, Streamlit, Google Cloud (Cloud Run, Artifact Registry, Cloud Storage, Secret Manager, Workload Identity Federation), Neon, Anthropic Claude API.

---

## Results

Test period: 1 September 2025 to 1 September 2026 (17,566 half-hours never seen in training).

### Demand forecast

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Naive baseline (same time last week) | 2,132 MW | 2,878 MW | 8.40% |
| **LightGBM** | **1,241 MW** | **1,620 MW** | **5.08%** |

| Day | MAPE |
|---|---|
| Typical weekday (26 August 2026) | 2.67% |
| Christmas Day 2025 | 13.52% |

### Demand prediction intervals

Three LightGBM quantile models (q10, q50, q90) give an 80% prediction interval around each demand forecast. A well calibrated 80% interval should contain about 80% of actual values, with about 10% falling below and 10% above.

| Intervals | Coverage | Below | Above | Average width |
|---|---|---|---|---|
| Raw quantile models | 51.0% | 20.7% | 28.3% | 2,448 MW |
| Conformally calibrated | 75.7% | 11.2% | 13.1% | 3,964 MW |
| **Calibrated, models retrained on full training period** | **80.2%** | **9.3%** | **8.8%** | **4,204 MW** |

The raw quantile models were overconfident, covering only half of actual values. Calibration widened each interval by 758 MW, bringing coverage to the 80% target with misses balanced on both sides. The intervals also adapt to the time of day: they are narrowest overnight (around 3,000 MW) and widest around midday (up to around 6,000 MW).

### Gradient boosting vs deep learning

Can deep learning beat gradient boosting here? Three kinds of deep learning model were compared with LightGBM on the same days: PatchTST, a transformer implemented from the paper in PyTorch and trained from scratch on this data; Chronos-Bolt and Chronos 2, used zero shot; and Chronos 2 fine tuned on this data. The diagram shows PatchTST.

```mermaid
flowchart TD
    week["Past week of demand<br/>336 half-hours, up to 08:00 the day before"] --> revin["RevIN: scale the week to mean 0, spread 1"]
    revin --> patches["27 overlapping 12-hour patches"]
    patches --> encoder["Transformer encoder<br/>2 layers, 4 heads"]
    inputs["Tomorrow's weather and calendar<br/>48 half hours"] --> covariates["Covariate encoder"]
    covariates --> query["One query per forecast half hour"]
    encoder --> cross["Cross attention: each half hour picks the patches it needs"]
    query --> cross
    cross --> head["Quantile head<br/>low ≤ median ≤ high"]
    covariates --> head
    head --> undo["Undo RevIN: back to MW"]
    undo --> out["Tomorrow: 48 half hours × 10%, 50%, 90%"]
```

Every model sees the same information (demand up to 08:00 the day before, plus tomorrows weather and calendar), gets the same conformal calibration from the calibration year, and is scored on the same 363 test days. Clock change days are skipped for all of them, which is why LightGBM scores 5.08% here rather than 5.03%.

| Model | MAE | RMSE | MAPE | 80% interval coverage | Average width |
|---|---|---|---|---|---|
| Naive baseline (same time last week) | 2,130 MW | 2,875 MW | 8.39% | — | — |
| Chronos-Bolt small (zero shot, demand only) | 1,839 MW | 2,437 MW | 7.29% | 78.6% | 5,614 MW |
| Chronos 2 (zero shot, demand only) | 1,673 MW | 2,266 MW | 6.72% | 79.9% | 5,242 MW |
| PatchTST (trained from scratch) | 1,368 MW | 1,793 MW | 5.61% | 77.7% | 4,135 MW |
| Chronos-2 (zero shot, with weather and calendar) | 1,383 MW | 1,861 MW | 5.53% | 81.4% | 4,757 MW |
| LightGBM (in production) | 1,246 MW | 1,647 MW | 5.08% | 80.2% | 4,035 MW |
| Blend: 75% LightGBM + 25% PatchTST | — | — | 4.97% | 79.2% | 3,796 MW |
| Blend: 50% LightGBM + 15% PatchTST + 35% zero shot Chronos 2 | — | — | 4.71% | 80.1% | 3,768 MW |
| Chronos 2 (fine-tuned, with weather and calendar) | 1,125 MW | 1,475 MW | 4.49% | 81.3% | 3,701 MW |
| **Blend: 35% LightGBM + 5% PatchTST + 60% fine-tuned Chronos-2** | — | — | **4.36%** | **80.4%** | **3,483 MW** |

![PatchTST and LightGBM forecasts for the week of 5 January 2026](docs/patchtst_forecast_week.png)

![Which past half hours each of tomorrow's forecasts depends on](docs/patchtst_influence.png)

- **Fine tuning a pretrained foundation model beat everything else.** Zero shot, with tomorrow's weather and calendar as known future inputs, Chronos 2 already matches PatchTST trained from scratch (5.53% vs 5.61%) without being trained on this data. Fine-tuning it for 500 steps on a free Colab GPU (`notebooks/chronos2_finetune_colab.ipynb`) cut its error by a further 19%, to 4.49%: 12% better than LightGBM, with narrower intervals that stay calibrated (81.3%). Pretraining on a large and varied set of time series gives it knowledge that three years of GB demand alone can't, and fine-tuning adapts that to this data. Fine-tuning only used data up to each forecast's 08:00 cutoff, Chronos-2 was released in October 2025 so most of the test year comes after it, and zero shot it doesn't beat LightGBM, so it isn't simply recognising this data. This is one fine-tuning run; runs with other random seeds are planned.
- **Weather matters to the pretrained model too.** On demand alone Chronos 2 scores 6.72%, 8% better than the older Chronos-Bolt; given the weather and calendar, 5.53%, a further 18% better.
- **The best forecast blends gradient boosting with deep learning.** 35% LightGBM, 5% PatchTST and 60% fine-tuned Chronos 2 scores 4.36%, 14% lower error than LightGBM alone, with 80.4% coverage and intervals 14% narrower. Every choice was made on the calibration year: the fine tuned Chronos 2 over the zero shot one (4.01% vs 4.30% calibration MAPE for their best blends), and the weights. Weights chosen with hindsight on the test year would only reach 4.34%, so the choice transferred well. LightGBM keeps a third of the weight because its mistakes differ from the deep models'. PatchTST's share falls to 5% because fine tuned Chronos 2 covers most of what it added: without it the blend scores 4.02% on the calibration year and the same 4.36% on the test year.
- **No other blend does meaningfully better.** Every blend of the five models with validation year forecasts (LightGBM, PatchTST and the three Chronos 2 versions), in steps of 5% (10,626 blends, `python src/ensemble.py --search-all`), was scored on the validation year. The best scored 4.007% there, against 4.01% for the three-way blend: a difference far below the variation between random seeds. The top blends form a flat plateau (all within 0.003 points), share the same shape (about 30 to 35% LightGBM, 55 to 60% fine tuned Chronos 2 and 5 to 10% PatchTST) and score 4.35 to 4.37% on the test year. The two zero-shot Chronos 2 versions get at most 10% between them, because their mistakes largely repeat the fine tuned version's. So the three-way blend is kept as the final model: no larger blend is meaningfully better, and every extra weight chosen on the same year adds risk of overfitting it.
- **PatchTST beats the pretrained Chronos-Bolt by 23% but not LightGBM alone.** This matches what forecasting competitions such as M5 found: gradient boosting with good features is hard to beat on a few years of data, for a model trained from scratch.
- **Letting each forecast half-hour choose what to look at helped most.** The papers head flattens the 27 patch summaries into one long vector, through a single layer holding 221,312 of the model's 347,525 parameters. Three heads were compared on the same days with `src/ablation.py`:

  | Head | Parameters | MAPE | 80% interval coverage | Average width |
  |---|---|---|---|---|
  | Flatten (paper) | 347,525 | 5.98% | 78.7% | 4,656 MW |
  | Mean of the patches | 134,533 | 5.88% | 79.4% | 4,538 MW |
  | **Cross attention** | **153,413** | **5.61%** | **77.7%** | **4,135 MW** |

  Averaging helps a little. Cross attention, where each of tomorrow's half hours builds a query from its time of day, weather and calendar and picks the patches it needs, cut the error by 6% and the interval width by 11% with 56% fewer parameters.

- **Attribution shows what changed.** The heatmap shows how much each of tomorrow's forecasts moves when each past half hour changes (the gradient of the forecast, averaged over every test day). With the flatten head, the stripes were vertical: every past half hour affected every forecast about equally, so the past week only set tomorrow's level. With cross attention, the influence peaks at the same times every day across the week and is stronger for daytime forecasts than overnight ones, so the transformer has learned to match times of day, as LightGBM does with its same time yesterday and same time last week features. The most recent full day matters most.
- **Giving it the matching past half-hours directly added nothing.** Feeding each forecast half-hour the demand at the same time 2 and 7 days earlier, LightGBM's strongest kind of feature, moved test MAPE from 5.61% to 5.60%. Cross attention had already learned to find those values itself, as the heatmap shows.
- **Tuning overfitted the calibration year, and made the transformer more like LightGBM.** Optuna searched 40 settings (learning rate, dropout, model size, patch length, lags), scored only on the calibration year, with median pruning. The winner, a smaller model with one layer and day long patches, cut calibration year error by 7%, but averaged over three random seeds it matched the default on the test year (5.61% ± 0.07 vs 5.58% ± 0.10), with 69% fewer parameters and better coverage (79.4%). Comparing 40 settings on one year picks partly the setting that suits that year. Its attribution map shows why it blends worse: it relies almost entirely on the demand 2 and 7 days before each half hour, as LightGBM does ([heatmap](docs/patchtst_influence_tuned.png)), so it repeats LightGBM's mistakes, such as under forecasting the first working day after New Year when both lag days fell in the holiday. The default settings were kept.
- **Blending still helps, but less than the transformer's own gain suggests.** 75% LightGBM and 25% PatchTST, with the weight chosen on the calibration year, gives 4.97% MAPE against 5.08% for LightGBM, with 80% intervals 6% narrower at 79.2% coverage. With the weaker flatten head the blend scored 4.98%: as PatchTST learned to match times of day, its errors became more like LightGBM's, so it improved alone more than it adds to the blend.
- **Training is noisy.** The calibration loss jumps between epochs and early stopping picks epoch 8 with every head ([learning curve](docs/patchtst_learning_curve.png)), and tuning didn't fix it, so results are compared over three random seeds.
- **Ablations show weather matters most, and RevIN keeps the intervals honest.** Each change was trained with three random seeds (`src/ablation.py`), and only differences clearly larger than the spread between seeds count:

  | Change from the default | Test MAPE (3 seeds) | 80% interval coverage | Average width |
  |---|---|---|---|
  | Default (RevIN, weather, 12 hour patches) | 5.64% ± 0.18 | 79.2% | 4,263 MW |
  | No weather, calendar only | 6.47% ± 0.07 | 81.5% | 5,192 MW |
  | No RevIN, one fixed scale | 5.56% ± 0.39 | 75.9% | 4,046 MW |
  | 6 hour patches | 5.83% ± 0.22 | 78.3% | 4,407 MW |
  | 24 hour patches | 5.63% ± 0.15 | 81.1% | 4,472 MW |

  Without tomorrows weather, error rises 15% and intervals widen 22%: temperature drives heating demand and sunshine drives rooftop solar. Removing RevIN leaves the average error unchanged but doubles the spread between seeds, and coverage falls to 75.9%: with one fixed scale the model can't adjust when demand levels shift, so intervals calibrated on one year are too narrow for the next. Patch length barely matters above 12 hours. Rerunning the default with the same seeds after a small code change moved its average from 5.58% to 5.64%, a reminder that single run differences under about 0.2 points are noise.

### Price forecast

The target is the Market Index Price, the half hourly price of electricity traded in the short term market. Prices can be negative, so models are scored on MAE and RMSE in £/MWh rather than MAPE.

| Model | MAE | RMSE |
|---|---|---|
| Naive baseline: same time last week | £27.57 | £41.25 |
| Naive baseline: most recent known day | £25.25 | £39.22 |
| **LightGBM** | **£21.50** | **£29.45** |

LightGBM is 15% better than the stronger baseline on MAE and 25% better on RMSE. The larger RMSE improvement means it mostly avoids the big misses the baselines make, by seeing windy, cheap periods and tight, expensive ones coming.

**Feature selection on a validation year.** The first price model relied heavily on `dayofyear`. Unlike demand, price levels drift between years with gas prices (around £150/MWh in early 2023, around £70 in 2024, rising again through 2026), so seasonal patterns learned from past years don't generalise. Comparing feature sets on a held-out validation year (September 2024 to August 2025), removing `dayofyear` improved MAE from £18.57 to £17.12 (7.8%). That version was chosen before the test set was evaluated, then tested once.

### Price spike classifier

A spike is defined relative to the recent price level: **a price more than £43.72/MWh above its own 7 day average**. The threshold was set so that 5% of training half-hours count as spikes, using training data only. The test year was more volatile, with 6.3% spikes.

The baseline is persistence: predict a spike if the most recent known price at that time was itself spike-level.

| Model | Precision | Recall | F1 | Average precision |
|---|---|---|---|---|
| Random guessing | — | — | — | 0.063 |
| Persistence | 0.39 | 0.33 | 0.36 | 0.170 |
| LightGBM, 0.5 cut-off | 0.75 | 0.12 | 0.21 | **0.428** |
| LightGBM, cut-off chosen on validation (0.30) | 0.59 | 0.19 | 0.29 | **0.428** |

ROC AUC: 0.899.

- **The classifier ranks risk far better than persistence.** Its average precision, which scores the probabilities across every possible cut off, is 2.5 times persistence's and nearly 7 times random guessing.
- **Its alarms are more precise:** 59% of half hours it flags are real spikes, against 39% for persistence.
- **At a single cut off it catches fewer spikes:** Persistence scores a slightly higher F1. The cut off was chosen on the validation year, and the more volatile test year needed a less cautious one. It was not retuned on the test set.
- In the battery backtest below, using the spike probabilities to adjust the price forecast added no profit. The classifier ranks risk well, but its signal doesn't change one cycle a day trading decisions.

**Feature choice:** Tree models split on one feature at a time, so they can't easily compute the difference between two features. Adding `price_excess_recent` (the most recent known price minus its 7 day average, persistence's own signal) raised validation average precision from 0.405 to 0.412 and was kept.

### Battery trading backtest

Does a more accurate forecast actually make more money? A simulated battery plans each day's charging and discharging from a price forecast, using linear programming to find the most profitable schedule, and the plan is then scored on the real prices.

**Battery:** 1 MW power, 2 MWh storage, 90% round-trip efficiency, at most one full cycle a day, each day starting empty.

| Strategy | Forecast used to plan | Profit over test year | Share of perfect foresight | Losing days |
|---|---|---|---|---|
| Perfect foresight | Real prices (upper limit) | £42,937 | 100.0% | 0 |
| LightGBM | Price model | **£33,095** | **77.1%** | 17 |
| LightGBM + spike model | Price model, raised to the spike level where spike probability passes the cut-off | £33,048 | 77.0% | 16 |
| 7-day average | Average price at the same half-hour over the last 7 known days | £31,872 | 74.2% | 24 |
| Recent day | Most recent known price at each half hour | £22,377 | 52.1% | 45 |

- **The forecast is worth real money:** about £10,700 a year more than the recent day forecast (+48%), with far fewer losing days.
- **Most of the value comes from the daily shape.** Simply averaging the last 7 days prices captures 74%. A battery only needs to know when prices are low and high, not their exact level, so LightGBM's accuracy advantage turns into only about £1,200 (4%) more profit than the 7 day average.
- **The spike model doesnt change the outcome.**
- **The LightGBM result is an upper bound.** The price model uses actual wind, solar and demand rather than forecasts, so with real inputs its small lead over the 7 day average could shrink. The 7 day average uses only past prices, so its result is fully realistic.

### Monitoring

The demand model was replayed over the test year day by day, as if each day had just arrived, and scored on accuracy, interval coverage and input drift (`src/monitor.py`, shown on the Streamlit dashboard).

- **Interval coverage** stayed near or above the 80% target for most of the year, but the 30 day coverage fell below 70% on 12 days, all over Christmas and New Year (lowest about 66% in mid January). Coverage holds over the year as a whole, not in every period.
- **Accuracy** was no better than the same time last week baseline (7 day average) on 26 days, in three spells: 2 January, 31 January to 9 February, and 18 July to 4 August. The July spell coincides with the largest input drift of the year: temperature in July 2026 differed sharply from the training years Julys (PSI 0.97), giving the model weather it had seen little of.
- **Input drift:** sunshine and recent demand stayed stable or moderate (PSI up to 0.13 and 0.20, the latter over Christmas), but temperature shifted by more than 0.25 in half the months, because weather comes in spells lasting days or weeks.

### Live forecasts

Each morning the daily job also asks Claude (Haiku, Anthropic API) to turn the day's figures, meaning tomorrow's peak and low, interval width, recent accuracy against the baseline, coverage and any alerts, into a short english briefing shown at the top of the dashboard. Before it is saved, every number in the briefing is checked against the figures Claude was given. If any number doesn't match (allowing for rounding), the briefing is rejected and the reason is logged, so a hallucinated figure can't reach the dashboard. If the API is unavailable, the forecast job carries on without a briefing.

Since October 2026, a scheduled job forecasts the next days demand every morning from the latest Elexon demand and an Open Meteo **weather forecast**, the inputs a real forecaster would have. Each forecast is scored once that days actual demand is published, with the same daily accuracy, baseline and coverage checks as the replay above, and shown on the live dashboard. The model is trained and tested on archived weather forecasts rather than measured weather, so the test results already include the effect of weather forecast errors, and the live scores check that this holds in practice.

### What the data shows about prices

- **Daily shape:** an evening peak at 6–7pm UK time (about £112/MWh on average), a smaller morning peak, an overnight dip (about £69), and a midday dip (about £74) caused by solar.
- **Fat tails:** the middle half of prices sits between £68 and £104/MWh, but prices ranged from −£102.92 to £1,352.90.
- **Negative prices** occurred in 1,684 half hours (2.6%), clustered around midday from solar and overnight on windy nights with low demand, and concentrated between April and September. They almost never occur during the evening peak.

### Error analysis

- **Solar:** errors peaked in the middle of the day, at 2.0–2.4 GW around midday compared with about 1 GW overnight. Rooftop solar isn't measured in this demand data, so sunny days reduce the demand the grid sees. Adding shortwave radiation as a feature reduced MAPE from 5.66% to 5.04%.
- **Christmas:** Christmas Day had a MAPE of 14.8%, with only two Christmases in the training data. A Christmas-period flag reduced this to 13.5%, but the model still overestimates demand.
- **Peaks:** the model gets the shape of each day right but underestimates the extremes, forecasting too low at the morning ramp and the evening peak.

---

## Quick start

The data and trained model aren't stored in the repository, so the pipeline has to be run once to create them.

**1. Set up and run the pipeline**

The pipeline stores its processed data in PostgreSQL, which runs in Docker.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

docker compose up -d db        # start PostgreSQL 
python src/ingest.py           # download demand and weather data
python src/validate.py         # check data quality
python src/preprocess.py       # clean, align and join
python src/features.py         # build model features
python src/train.py            # train and compare with the baseline
python src/quantiles.py        # train and calibrate the prediction intervals
python src/price_features.py   # build price features
python src/train_price.py      # select features, train the price model, compare with baselines
python src/spikes.py           # train the spike classifier and choose its cut off
python src/battery_backtest.py # backtest battery trading strategies and save the results for the dashboard
python src/monitor.py          # replay the test year and save the monitoring tables
python src/live.py             # forecast tomorrow from live data and score earlier live forecasts
Set `ANTHROPIC_API_KEY` to also write the daily Claude briefing. Without it, `live.py` skips the briefing and runs as normal.
```
**Transformer experiments** These need PyTorch, which the deployed API doesn't use, so it's kept out of `requirements.txt` and the Docker image:

```bash
pip install -r requirements-research.txt
python src/train_patchtst.py   # train PatchTST and compare it with LightGBM and Chronos on the same days
python src/plot_patchtst.py    # draw the figures in docs/
python src/ensemble.py         # blend LightGBM and PatchTST, choosing the weight on the calibration year
python src/ablation.py pooling flatten mean attention   # compare settings, here the three heads
python src/ablation.py weather 1 0 --seeds 42 1 2   # e.g. retrain with and without weather, three seeds each
python src/tune_patchtst.py --trials 40   # tune PatchTST with Optuna on the calibration year only
python src/chronos2.py --skip-finetune    # Chronos 2 zero shot, with and without weather (run after train_patchtst.py)
python src/chronos2.py --export data/chronos2_input.csv.gz    # inputs for fine tuning on a GPU, see notebooks/chronos2_finetune_colab.ipynb
python src/chronos2.py --import data/chronos2_finetuned.csv.gz   # add the fine-tuned forecasts made on the GPU
python src/ensemble.py --search-all       # two and three way blends, plus every blend of every model
```

Each training script logs its settings, results and model to MLflow. Open `http://localhost:5001` to browse and compare runs, and to see model versions.
On macOS, LightGBM needs OpenMP first: `brew install libomp`

**2. Serve forecasts and the dashboard**, with Docker Compose:

```bash
docker compose up --build
```

This starts the database, the API (`http://localhost:8000/docs`), the monitoring dashboard (`http://localhost:8501`) and MLflow (`http://localhost:5001`).

or locally, with the database already running:

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive API documentation.

**3. Run the tests**

```bash
pytest
ruff check .
```

Every push runs CI on GitHub Actions; Ruff linting, a PostgreSQL database started with Docker Compose, the test suite with coverage, and the API container started with Docker Compose and checked by requesting a forecast. In CI, the tests run against a small model and features table built from synthetic data, since the real data and models aren't stored in the repository. The script that creates them refuses to run outside CI, so it can't overwrite real models. Dependabot opens weekly pull requests for dependency updates.

When CI passes on `main`, the deploy workflow builds the image with the current models, pushes it to artifact registry tagged with the commit ID, deploys the API and dashboard to cloud run, and checks the live API returns a forecast.

---

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Returns `{"status": "ok"}` if the service is running |
| `GET /predict?day=YYYY-MM-DD` | Half-hourly forecast, 80% preditcion interval and actual demand for that day, with the day's MAE and MAPE |
| `GET /forecast/latest` | The newest live forecast (normally tomorrow), with its 80% interval; returns 404 until the first daily run |

Example response (trimmed):

```json
{
  "date": "2026-08-26",
  "in_training_data": false,
  "mae": 668,
  "mape": 2.67,
  "interval": "80%",
  "forecast": [
    {"timestamp_utc": "2026-08-25T23:00:00", "low": 20175, "forecast": 21673, "high": 22963, "actual": 22284}
  ]
}
```

`in_training_data` is `true` when the requested day was part of the training set, where results look better than the model's real performance. Invalid dates return 422, and dates outside the data return 404.

A single day can also be forecast from the command line: `python src/predict.py 2026-08-26`

---

## How it works

### Deployment

```mermaid
flowchart LR
    push["Push to main"] --> ci["CI: lint, tests, Docker Compose checks"]
    ci -->|passes| deploy["Deploy: build image, push to Artifact Registry"]
    bucket[("Cloud Storage: models")] --> deploy
    deploy --> api["Cloud Run: API"]
    deploy --> dash["Cloud Run: dashboard"]
    retrain["Weekly retrain"] -->|passes quality gate| bucket
    retrain --> deploy
    bucket --> daily["Daily forecast job"]
    daily -->|figures| claude["Claude API"]
    claude -->|briefing, number-checked| daily
    daily --> db[("Neon PostgreSQL")]
    api --> db
    dash --> db
```

### Pipeline

| Stage | File | Purpose |
|---|---|---|
| Ingest | `src/ingest.py` | Downloads demand from Elexon in weekly chunks and weather from Open Meteo, retrying with increasing waits when an API is briefly unavailable |
| Validate | `src/validate.py` | Checks for missing periods, duplicates, nulls, out of range values and date coverage |
| Explore | `notebooks/exploration.ipynb` | Daily, weekly and seasonal patterns, temperature relationship, autocorrelation, holidays |
| Preprocess | `src/preprocess.py` | Converts to UTC, fills gaps, resamples weather to half hourly, joins the sources and saves them to PostgreSQL |
| Database | `src/db.py` | Saves and loads tables in PostgreSQL; every stage after preprocessing reads and writes through it |
| Features | `src/features.py` | Builds calendar, weather and lag features |
| Train | `src/train.py` | Trains LightGBM with a time based split and compares it with the baseline |
| Quantiles | `src/quantiles.py` | Trains q10/q50/q90 models and calibrates the prediction intervals |
| PatchTST | `src/patchtst.py`, `src/forecast_windows.py` | The transformer (RevIN, patching, encoder, quantile head) and its training loop, and the forecast windows every model is scored on |
| Transformer comparison | `src/train_patchtst.py` | Trains PatchTST with early stopping, calibrates it, and compares it with LightGBM and Chronos on the same test days |
| Transformer figures | `src/plot_patchtst.py` | Draws a forecast week, the learning curve and the attribution heatmap into `docs/` |
| Ensemble and ablations | `src/ensemble.py`, `src/ablation.py` | Blends LightGBM, PatchTST and Chronos 2 with weights chosen on the calibration year, searches every blend of the models, and retrains PatchTST once per value of a setting over several seeds |
| Tuning | `src/tune_patchtst.py` | Tunes PatchTSTs settings with Optuna, scored on the calibration year only |
| Chronos-2 | `src/chronos2.py`, `notebooks/chronos2_finetune_colab.ipynb` | Forecasts with Chronos 2 zero shot (with and without weather and calendar) and fine tuned, with export and import for fine tuning on a cloud GPU |
| Retrain | `src/retrain.py` | Checks a newly trained model against the baseline on the latest 8 weeks, then retrains the demand and interval models on all data and recalibrates the intervals |
| Price features | `src/price_features.py` | Adds wind, solar, net demand and price lag features |
| Price model | `src/train_price.py` | Selects features on a validation year, trains LightGBM and compares it with two baselines |
| Spikes | `src/spikes.py` | Labels price spikes, selects features and a probability cut-off on a validation year, and compares the classifier with persistence |
| Backtest | `src/battery_backtest.py` | Plans each day's battery schedule from a price forecast with linear programming and scores it on real prices over the test year |
| Monitor | `src/monitor.py` | Replays the test year day by day: daily error against the baseline, interval coverage and monthly input drift, saved to PostgreSQL |
| Live | `src/live.py` | Fetches the last 21 days of demand and a weather forecast, forecasts tomorrow, and scores earlier live forecasts once their actual demand is known |
| Briefing | `src/briefing.py` | Sends the day's figures to Claude for a short english briefing, and rejects it if it contains any number not in those figures |
| Dashboard | `dashboard/app.py` | Streamlit dashboard in two sections. Demand forecasting: a live page using LightGBM (forecasts, accuracy, coverage, drift and freshness, with alerts) and a test year page comparing gradient boosting with deep learning (sliders to try any blend of LightGBM, PatchTST and fine tuned Chronos 2, and each model's error by month). Prices and trading, using LightGBM: the price forecast, spike classifier and battery backtest on the test year |
| Evaluate | `src/evaluate.py` | MAE, RMSE, MAPE and quantile loss |
| Tracking | `src/tracking.py` | Logs each training runs settings, metrics and model files to MLflow, and registers the demand, price and spike models |
| Predict | `src/predict.py` | Loads the saved models and forecasts a chosen day with interval |
| Serve | `api/main.py` | FastAPI service, packaged with the `Dockerfile` and run alongside PostgreSQL with `docker-compose.yml` |
| Test | `tests/` | Unit tests for metrics, features, the spike definition, the battery, the monitoring, the database, the briefing (using a mocked Claude client), the transformer (no look ahead in its windows, non crossing quantiles, reproducibility, saving and loading), Chronos 2 (inputs stop at the 08:00 cutoff, forecasts made on another machine are matched to the right days) and the blends, plus API tests |
| CI | `.github/workflows/tests.yml` | Lints, starts PostgreSQL, runs the tests with coverage, and starts the API and dashboard with Docker Compose on every push |
| Scheduled retraining | `.github/workflows/retrain.yml` | Every Monday, downloads the latest data, rebuilds the features and runs the retraining, publishing if the quality gate is passed |
| Deploy | `.github/workflows/deploy.yml` | After CI passes on `main`, or after a retrain, builds and pushes the image and deploys the API and dashboard to cloud run |
| Daily forecast | `.github/workflows/daily.yml` | Every day at 10:00 UTC, runs `src/live.py` against the hosted database |

### Data

- **Demand:** Initial national demand outturn from the Elexon BMRS API, half hourly, January 2023 to September 2026 (64,316 settlement periods).
- **Weather:** Hourly temperature, wind speed and shortwave radiation for London from the Open Meteo historical forecast API, interpolated to half hourly. These are archived weather forecasts rather than measured weather, so the model trains on the same kind of input it gets live, where the daily job uses the Open Meteo forecast API. Training on measured weather and serving on forecasts would make the model rely on weather inputs more than live forecasts can support.
- **Prices:** Market Index Price (APX/EPEX) from the Elexon BMRS API. Six missing half hours and 34 half hours with zero traded volume were filled by interpolation.
- **Wind and solar generation:** Actual onshore wind, offshore wind and solar generation from the Elexon BMRS API. 4,383 republished duplicate rows were removed (keeping the latest version), 863 missing half-hours (longest gap 7 hours) were filled by interpolation, and 15 small negative values were set to zero.
- **Storage:** Raw downloads are kept as CSV files, untouched. Everything from cleaning onwards is stored in PostgreSQL tables: `dataset`, `market`, `features` and `price_features`, plus `monitoring_daily` and `monitoring_drift` for the replay, `live_forecasts`, `live_briefing` and `monitoring_live` for live forecasts, `backtest_daily`, `price_forecasts`, `transformer_forecasts`, `transformer_calibration`, `transformer_training`, `transformer_influence`, `ensemble_weights`, `ensemble_search` and `transformer_ablation` for the research pages.

Validation found two days each missing one settlement period (2023-07-17 period 45 and 2023-12-29 period 8), which were filled by time interpolation.

### Features

| Feature | Description |
|---|---|
| `period` | Settlement period (1–48), in UK local time |
| `dayofweek`, `dayofyear` | Calendar position |
| `is_holiday` | England and Wales bank holidays |
| `is_christmas` | 24 December to 1 January |
| `temperature_2m` | Air temperature (°C) |
| `hdd` | Heating degree days: max(0, 15.5 − temperature) |
| `shortwave_radiation` | Sunlight reaching the ground (W/m²), a proxy for rooftop solar output |
| `demand_lag_recent` | Demand at the same time yesterday, or two days ago (see below) |
| `demand_lag_336` | Demand at the same time last week |
| `demand_roll_96` | 24 hour average demand ending two days earlier |

Wind speed was dropped because it had almost no correlation with demand (0.03).

### Price features

The price model uses the demand, weather and calendar features (without `dayofyear`) plus:

| Feature | Description |
|---|---|
| `wind_onshore`, `wind_offshore`, `wind_total` | Wind generation (MW) |
| `solar_generation` | Solar generation (MW) |
| `net_demand` | Demand minus wind and solar: what's left to be met mostly by gas, which usually sets the price |
| `price_lag_recent` | Price at the same time yesterday, or two days ago (same rule as demand) |
| `price_lag_336` | Price at the same time last week |
| `price_roll_7d` | 7 day average price ending two days earlier, which tells the model the current price level |

### Design decisions

**A realistic day-ahead forecast.** The forecast for a given day is assumed to be made around 9am the day before, in line with GB day ahead power auctions, which run in the morning. At that point, yesterday's demand is only known up to about 8am. So `demand_lag_recent` uses lag 48 (yesterday) for periods before 8am local time and lag 96 (two days ago) from 8am onwards. Using lag 48 everywhere would quietly use data a real forecaster wouldn't have yet. A unit test enforces this rule, and the rolling average is shifted by 96 periods for the same reason.

**Time-based split.** Training data ends on 31 August 2025 and testing starts on 1 September 2025, so the model is never trained on data from after the period it's tested on.

**Conformal calibration of the intervals.** Quantile models learn their spread from training data they fit closely, so their intervals come out too narrow on new data. To correct this, the quantile models were first trained on data up to August 2024 and evaluated on a held out calibration year (September 2024 to August 2025). The amount by which actual values fell outside their intervals gave a 758 MW adjustment, the widening needed for 80% of calibration values to fall inside. The final models were retrained on the full training period and the same adjustment applied.

**Timezones.** Timestamps are stored in UTC to avoid duplicate and missing hours at the clock changes. Calendar features use UK local time, because that's when people actually use electricity.

**Net demand for prices.** Prices are set by supply and demand together. Low demand alone doesn't explain negative prices (overnight demand is low every night, but prices go negative on only a small share of nights), whereas low demand combined with high wind or solar does. Net demand captures this directly.

**Separate market data.** Prices and generation are stored in their own table, so adding them didn't change the demand pipeline or its results.

**Spikes relative to the recent level.** Price levels drift by a factor of two between years, so a fixed threshold such as "above £150/MWh" would label most of early 2023 as spikes and almost nothing in 2024, and the classifier would simply learn which periods were expensive. Defining a spike as a jump above the recent 7 day average captures sudden, unusual prices whatever the current level.

**Why demand is the live system and prices are test year studies.** Demand is forecast live, retrained weekly, monitored and used for the gradient boosting vs deep learning comparison, while the price forecast, spike classifier and battery backtest are evaluated on the test year only. Everything the demand model uses is known at 08:00 the day before (past demand, the calendar and weather forecasts, which it was also trained on), so its test year accuracy is a fair estimate of live accuracy. The price model uses the actual wind, solar and demand for the hours it forecasts, which aren't known in advance, so its results are an upper bound: running it live would mean retraining it on day ahead forecasts of those inputs, which would likely shrink its small lead over a 7 day average. More importantly, the price side is limited by its data, not its model. Gas usually sets GB power prices and isn't an input, and the extremes come from outages and scarcity the inputs can't see. Spikes are rare and their causes aren't in the features, and the battery only needs to know which hours are cheapest and dearest, which the daily price shape already gives: a 7 day average earns 74% of the perfect-foresight profit against 77% for LightGBM. So neither a live price system nor a deep learning comparison on prices would add much. The comparison belongs on demand, where the signal is strong enough for model differences to show, and the one change likely to help the spike model is adding the missing information: outage notices extracted from REMIT messages.

**PostgreSQL for processed data.** Raw downloads stay as CSV files so everything can be rebuilt without calling the APIs again, but every processed table lives in one place, the database, so training, backtesting and the API always read the same data. The connection comes from a `DATABASE_URL` environment variable, so the same code runs on a laptop, inside Docker Compose and in CI.

**Scheduled retraining with a quality gate.** Demand patterns drift as solar capacity grows and behaviour changes, so the models are retrained weekly on the latest data. Before anything is saved, a model trained without the most recent 8 weeks must beat the same time last week baseline on them, otherwise the run fails and the previous models are kept. Requests to Elexon and Open Meteo are retried up to five times, waiting longer each time, when the service is briefly down or rate limiting (a 503 from Open Meteo once failed a scheduled run), so a short outage doesn't stop the retrain. The interval calibration is recomputed on the same recent weeks. The READMEs results use data up to 1 September 2026 so they stay reproducible, only the retraining workflow fetches newer data.

**Experiment tracking with MLflow.** Every training run records its settings, metrics and model files in MLflow, so results can be compared across runs and any past model can be recovered exactly. The demand, price and spike models are registered as numbered versions. The MLflow server runs in Docker Compose, storing its records in a separate PostgreSQL database and its model files in a Docker volume.

**Monitoring alerts on performance, not on drift alone.** Input drift is measured with the population stability index against the same calendar month in the training years, because comparing a summer month with a whole training year would flag every season as drift. Even so, temperature shifts by more than 0.25 in half the months, since a single month holds only a handful of weather spells, so a drift alert would fire constantly and be ignored. Alerts are therefore based on accuracy (7 day error no better than the baseline), interval coverage (30 day coverage below 70%) and data freshness, and drift is shown as a likely explanation when accuracy drops, as it did in July 2026.

**Serverless deployment on Google Cloud.** The API and dashboard run on cloud run from the same image, so they scale to zero when idle and cost almost nothing. The first request after a quiet spell takes 10–20 seconds while a container starts and loads the data. The database is neons hosted PostgreSQL in London, next to the cloud run region, because Googles own Cloud SQL has no free tier. The database URL is kept in Secret Manager, not in code or in the image.

**Keyless, gated continuous deployment.** GitHub actions logs in to google cloud with workload identity federation: google trusts GitHubs tokens, but only from this repository, so no long lived key is stored anywhere. A dedicated service account can only deploy to cloud run, push to this projects registry and read and write the models bucket. Deploys happen only after CI passes, and each image is tagged with its commit, so any live version can be traced or rolled back. Models live in a cloud storage bucket rather than in git, and a retrained model reaches production only if it passes the quality gate.

**Grounded LLM briefings.** Language models can write convincing numbers that are wrong, which matters in a forecasting system. Claude is given only a short list of facts computed by the pipeline and told to copy numbers exactly, and the code then extracts every number from its reply and checks each one against those facts, allowing for rounding. A reply with any unmatched number is discarded. The model never sees raw data or produces forecasts itself, so it explains results rather than creating them. The API key lives in a GitHub Actions secret, the key is scoped to a single workspace, and the tests use a mocked client, so CI runs without a key or any cost.

**Transformer comparison.** PatchTST sees exactly what the live job has at 08:00 the day before: the previous 7 days of demand, then a 16 hour gap, then tomorrow's weather and calendar. RevIN (reversible instance normalisation) scales each week to mean 0 and spread 1 and undoes it on the output, so the model learns the shape of demand and copes as the overall level drifts between seasons and years. The quantile head predicts the median plus two positive gaps, so its 10%, 50% and 90% forecasts can never cross. It's trained with AdamW, a warmup then cosine learning rate schedule, gradient clipping and early stopping on the calibration year, then refitted on the full training period for the best number of epochs, mirroring how the LightGBM intervals are built. On macOS, LightGBM and PyTorch each load their own OpenMP threading library and freeze or crash when both run in one process, so the comparison scores LightGBM in a separate process.

**Chronos 2.** Unlike Chronos-Bolt, Chronos 2 reads covariates, so it gets the same inputs as the other models: demand up to 08:00 the day before (about three weeks of it), and the weather and calendar for the 16 hour gap and tomorrow as known future values. It forecasts the 80 half hours from 08:00 and keeps the last 48. Fine tuning mirrors PatchTST: once on data before the calibration year, whose forecasts set the conformal adjustment and the blend weights, then again on everything before the test year for the test forecasts, each stopping at its first forecast's 08:00 cutoff. A 120M parameter model needs a GPU (on a MacBook Air one step took over four minutes), so fine-tuning runs on Google Colab's free T4 GPU from an exported file, using gradient accumulation to fit in memory, and the forecasts are imported back for calibration and scoring.

---

## Limitations

- The archived weather forecasts are stitched together from the first hours of each forecast run, which are more accurate than the day ahead forecasts available when a real forecast is made, so the test results may still slightly overstate live accuracy. Live forecasts are scored separately on the dashboard to measure this.
- Weather comes from London only, while national demand depends on weather across Great Britain, especially for solar.
- Holidays follow the England and Wales calendar; Scotland and Northern Ireland differ.
- MLflow runs locally only, so the weekly retraining on githubs machines isn't tracked there. Retrained models are versioned by their report in cloud storage and kept as 30 day workflow artifacts instead.
- The price, spike and battery models are evaluated on the test year only, with actual wind, solar and demand as inputs, so their results are an upper bound (see "Why demand is the live system and prices are test year studies" above).
- Interval coverage holds over the test year as a whole, not on every day, so easy days are covered more often and unusual days (such as Christmas) less often. The choice to apply the calibration to retrained models was checked once against the test set.
- The price model has no gas price input, although gas usually sets GB power prices. The 7-day average price captures its effect only indirectly.
- Half hourly prices are noisy, and extreme spikes and negative prices remain hard to predict.
- A spike probability cut off chosen on one year transfers imperfectly to a more volatile year, in practice it would need recalibrating regularly.
- The briefing check verifies that every number came from the source figures, but not that it's attached to the right label (a real MAPE figure described as coverage would pass), and it can't check numbers written as words.
- The fine tuned Chronos 2 result comes from one fine tuning run; more runs with other random seeds would show its spread. Fine tuning needs a GPU, so unlike LightGBM it isn't retrained automatically yet, and the blend isn't served live: LightGBM still makes the live forecasts. PatchTST uses its default settings, because tuning with Optuna didn't carry over to the test year. Chronos-Bolt is asked for 80 steps ahead (the 16 hour gap plus tomorrow), beyond the 64 it was trained for, which may understate it.

---

## Data sources and licences

- Contains BMRS data © Elexon Limited copyright and database right 2026. [Licence](https://www.elexon.co.uk/bsc/data/balancing-mechanism-reporting-agent/copyright-licence-bmrs-data/)
- Weather data by [Open-Meteo.com](https://open-meteo.com/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
