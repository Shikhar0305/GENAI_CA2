# Credit Card Fraud Detection API

This project trains a Random Forest model to detect fraudulent credit-card
transactions and exposes the saved model through a FastAPI service.

## Project files

- `creditcardfraud.py` - trains, evaluates, saves, and verifies the model.
- `fraud_api.py` - FastAPI application for fraud predictions.
- `test_fraud_api.py` - model and API tests.
- `Dockerfile` - container image definition for the API.
- `.github/workflows/ci.yml` - GitHub Actions test pipeline.

## Train the model

Place the Kaggle credit-card dataset in the project directory as
`creditcard.csv`, or set `DATA_PATH` to its location.

```powershell
$env:DATA_PATH="C:\path\to\creditcard.csv"
python creditcardfraud.py
```

Training creates `model.joblib`, which contains the trained model, feature
names, decision threshold, hyperparameters, and evaluation metrics.

## Run locally

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn fraud_api:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at
<http://127.0.0.1:8000/docs>.

### Health check

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Prediction

The `transaction` object must contain exactly the feature names stored in
`model.joblib`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary '{\"transaction\":{\"V1\":0.8,\"V2\":0.7,\"Amount\":10.0}}'
```

Example response:

```json
{
  "fraud_probability": 0.98,
  "prediction": 1,
  "threshold": 0.5
}
```

`prediction` is `1` for fraud and `0` for legitimate transactions.

## Run with Docker

The model must be trained first so that `model.joblib` exists:

```powershell
docker build -t credit-card-fraud-api .
docker run --rm -p 8000:8000 credit-card-fraud-api
```

Then use the health and prediction requests shown above.

## Run tests

```powershell
python -m pytest -q
```

The tests create a temporary model artifact and do not require the full
`creditcard.csv` dataset.

## Continuous integration

Every push to `main` and every pull request targeting `main` runs
`.github/workflows/ci.yml`. The workflow installs the dependencies with
Python 3.13 and runs the test suite.
