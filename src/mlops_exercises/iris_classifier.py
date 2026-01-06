import pickle
from pathlib import Path

import typer
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Main app
app = typer.Typer()

# Nested app for train subcommands
train_app = typer.Typer()


@train_app.command()
def svm(
    kernel: str = typer.Option("linear", "--kernel", help="Kernel type for SVM (linear, rbf, poly, sigmoid)"),
    output: str = typer.Option("svm_model.ckpt", "--output", "-o", help="Path to save the trained model"),
):
    """Train a Support Vector Machine (SVM) model."""
    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train a Support Vector Machine (SVM) model
    print(f"Training SVM with kernel='{kernel}'...")
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print("Training completed!")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Save the model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"\nModel saved to {output_path}")


@train_app.command()
def knn(
    n_neighbors: int = typer.Option(5, "--n-neighbors", help="Number of neighbors to use"),
    output: str = typer.Option("knn_model.ckpt", "--output", "-o", help="Path to save the trained model"),
):
    """Train a K-Nearest Neighbors (KNN) model."""
    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train a K-Nearest Neighbors model
    print(f"Training KNN with n_neighbors={n_neighbors}...")
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print("Training completed!")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    # Save the model
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"\nModel saved to {output_path}")


# Add the train app to the main app
app.add_typer(train_app, name="train", help="Train machine learning models")


@app.command()
def evaluate(model_path: str = typer.Argument(..., help="Path to the saved model file")):
    """Load a saved model and evaluate it on test data."""
    # Check if model file exists
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"Error: Model file '{model_path}' not found!")
        raise typer.Exit(code=1)

    # Load the model
    print(f"Loading model from {model_path}...")
    with open(model_file, "rb") as f:
        saved_data = pickle.load(f)

    model = saved_data["model"]
    scaler = saved_data["scaler"]

    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset (using same random_state to get same test set)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features using the loaded scaler
    x_test = scaler.transform(x_test)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Print the results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)


# this "if"-block is added to enable the script to be run from the command line
if __name__ == "__main__":
    app()
