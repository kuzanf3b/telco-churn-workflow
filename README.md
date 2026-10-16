# Telco Churn MLflow CI

An end-to-end machine learning workflow for training and packaging a Telco Customer Churn classification model with [MLflow](https://mlflow.org/), GitHub Actions, and Docker.

The project trains a Logistic Regression baseline model, records parameters, metrics, artifacts, and the model with MLflow, then builds and publishes an MLflow model image to Docker Hub through CI/CD.

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── main.yml                         # GitHub Actions CI/CD workflow
├── MLproject/
│   ├── MLproject                            # MLflow project definition
│   ├── conda.yaml                           # Reproducible MLflow environment
│   ├── modelling.py                         # Model training and evaluation script
│   ├── tautan-dockerhub.txt                 # Docker Hub repository link
│   └── telco_customer_churn_preprocessing/
│       ├── telco_churn_train.csv            # Preprocessed training data
│       ├── telco_churn_test.csv             # Preprocessed test data
│       └── artifacts/                        # Generated evaluation artifacts
└── .gitignore
```

## Machine Learning Workflow

The training workflow performs the following steps:

1. Loads the preprocessed Telco Customer Churn training and test datasets.
2. Separates the `Churn` target column from the input features.
3. Trains a Logistic Regression classification model.
4. Logs the model, parameters, metrics, and input signature with MLflow.
5. Generates evaluation artifacts:
   - Confusion matrix image
   - Classification report in JSON format
   - Evaluation metrics in JSON format
6. Stores the MLflow tracking data in the local `MLproject/mlruns` directory.
7. Builds a Docker image containing the MLflow model.
8. Pushes both the commit-specific image tag and the `latest` tag to Docker Hub.

The default model parameters are:

| Parameter | Default |
|---|---:|
| `max_iter` | `1000` |
| `C` | `1.0` |
| Solver | `liblinear` |
| Random state | `42` |

## Requirements

- Python `3.12.7`
- MLflow `2.19.0`
- pandas `2.3.3`
- NumPy `2.5.3`
- scikit-learn `1.5.2`
- SciPy `1.18.1`
- Matplotlib `3.10.0`
- cloudpickle `3.1.2`

All dependencies are declared in [`MLproject/conda.yaml`](MLproject/conda.yaml).

## Run Locally

From the repository root, run the MLflow project with:

```bash
cd MLproject
mlflow run . --env-manager=local
```

To provide custom Logistic Regression parameters:

```bash
cd MLproject
mlflow run . --env-manager=local -P max_iter=1500 -P C=0.5
```

You can also run the training script directly:

```bash
cd MLproject
python modelling.py --project-root .
```

The script expects these default dataset paths:

- `MLproject/telco_customer_churn_preprocessing/telco_churn_train.csv`
- `MLproject/telco_customer_churn_preprocessing/telco_churn_test.csv`

Optional command-line arguments are available for custom data paths, tracking URI, experiment name, output directory, `max_iter`, and `C`. Run `python modelling.py --help` for the complete list.

## MLflow Outputs

Training outputs are written to the following locations:

- `MLproject/mlruns`: Local MLflow tracking data and logged model runs
- `MLproject/telco_customer_churn_preprocessing/artifacts/confusion_matrix.png`: Confusion matrix
- `MLproject/telco_customer_churn_preprocessing/artifacts/classification_report.json`: Classification report
- `MLproject/telco_customer_churn_preprocessing/artifacts/evaluation_metrics.json`: Accuracy, precision, recall, F1, and ROC-AUC metrics

To start the MLflow UI locally:

```bash
cd MLproject
mlflow ui --backend-store-uri ./mlruns
```

Then open the URL displayed by MLflow in your browser.

## Continuous Integration and Docker Publishing

The GitHub Actions workflow in `.github/workflows/main.yml` runs on:

- Pushes to the `main` branch
- Pull requests targeting the `main` branch
- Manual workflow dispatches

The workflow uses Python `3.12.7`, installs the pinned dependencies, runs the MLflow project, and uploads MLflow artifacts to GitHub Actions with a 30-day retention period.

For non-pull-request events, the workflow also builds and publishes the Docker model image using these tags:

```text
kuzanf3b/telco-churn-mlflow:<commit-sha>
kuzanf3b/telco-churn-mlflow:latest
```

### Required GitHub Secrets

Configure the following repository secrets before enabling Docker publishing:

- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

## Docker Hub Image Repository

View the generated model images and available tags on Docker Hub:

[Open the Telco Churn MLflow Docker Hub repository](https://hub.docker.com/repository/docker/kuzanf3b/telco-churn-mlflow)

To pull the latest image:

```bash
docker pull kuzanf3b/telco-churn-mlflow:latest
```

To run the MLflow model serving container locally:

```bash
docker run --rm -p 5000:8080 kuzanf3b/telco-churn-mlflow:latest
```

The exact serving command and port may vary depending on the MLflow model-serving image generated by the installed MLflow version.

## Reproducibility

The workflow pins the Python and package versions used for training and packaging. MLflow artifacts are uploaded for each workflow run, while Docker images are tagged with the source commit SHA so that a model image can be traced back to the code revision that produced it.
