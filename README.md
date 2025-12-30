# mlops_exercises

DTU MLOps course exercises

## Usage

The project implements a CNN for corrupted MNIST digit classification with training, evaluation, and visualization capabilities.

### Model Architecture

Inspect the model architecture and parameter count:

```bash
python src/mlops_exercises/model.py
```

This prints the complete network structure and total trainable parameters (approximately 1.7M parameters).

### Training

Train the model on corrupted MNIST data:

```bash
python src/mlops_exercises/train.py train --lr 1e-3 --batch-size 32 --epochs 10
```

The training script saves the model weights to `models/model.pth` and training statistics plots to `reports/figures/training_statistics.png`. All hyperparameters (learning rate, batch size, epochs) are configurable via command line arguments.

### Evaluation

Evaluate a trained model on the test set:

```bash
python src/mlops_exercises/evaluate.py evaluate
```

By default loads the model from `models/model.pth`. Specify a different checkpoint using `--model-checkpoint path/to/model.pth`. Prints test accuracy and total correct predictions.

### Visualization

The visualization module provides three analysis tools accessible via subcommands.

#### Generate All Visualizations

Generate all three visualizations at once:

```bash
python src/mlops_exercises/visualize.py all
```

This runs explore, confusion, and embeddings sequentially, creating all visualization outputs in one command.

#### Explore Data Distribution

Visualize dataset balance and sample diversity:

```bash
python src/mlops_exercises/visualize.py explore
```

Generates a dual-panel figure showing class distribution statistics (train/test split) and a 10×10 grid of representative samples from each digit class. Output saved to `reports/figures/class_distribution.png`.

#### Confusion Matrix Analysis

Analyze model errors with confusion matrix:

```bash
python src/mlops_exercises/visualize.py confusion
```

Creates a normalized confusion matrix showing classification patterns and displays misclassified examples for error analysis. Useful for identifying systematic model failures and commonly confused digits. Output saved to `reports/figures/confusion_matrix.png`.

#### t-SNE Embeddings

Generate t-SNE embeddings visualization:

```bash
python src/mlops_exercises/visualize.py embeddings
```

Extracts 256-dimensional feature representations from the penultimate layer, applies t-SNE dimensionality reduction, and creates a 2D scatter plot colored by digit class. Reveals how the model separates different digits in feature space. Output saved to `reports/figures/embeddings.png`.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
