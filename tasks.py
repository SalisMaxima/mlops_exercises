import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_exercises"
PYTHON_VERSION = "3.12"


# Environment commands
@task
def bootstrap(ctx: Context, name: str = ".venv") -> None:
    """Bootstrap a UV virtual environment and install dependencies."""
    ctx.run(f"uv venv {name}", echo=True, pty=not WINDOWS)
    ctx.run("uv sync", echo=True, pty=not WINDOWS)
    print(f"\n✓ Environment created at {name}")
    print(f"To activate: source {name}/bin/activate  (or {name}\\Scripts\\activate on Windows)")


@task
def sync(ctx: Context) -> None:
    """Install/sync all dependencies."""
    ctx.run("uv sync", echo=True, pty=not WINDOWS)


@task
def dev(ctx: Context) -> None:
    """Install with dev dependencies."""
    ctx.run("uv sync --dev", echo=True, pty=not WINDOWS)


# Check python path and version
@task
def python(ctx):
    """ """
    ctx.run("which python" if os.name != "nt" else "where python")
    ctx.run("python --version")


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, args: str = "") -> None:
    """Train model.

    Args:
        args: Hydra config overrides (e.g., "training.epochs=5 model.dropout=0.4")

    Examples:
        invoke train
        invoke train --args "training.epochs=5"
        invoke train --args "optimizer.lr=0.01 model.dropout=0.3"
    """
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py {args}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


# Code quality commands
@task
def ruff(ctx: Context) -> None:
    """Run ruff check and format."""
    ctx.run("uv run ruff check .", echo=True, pty=not WINDOWS)
    ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


# Git commands
@task
def git_status(ctx: Context) -> None:
    """Show git status."""
    ctx.run("git status", echo=True, pty=not WINDOWS)


@task
def git(ctx: Context, message: str) -> None:
    """Commit and push changes to git."""
    ctx.run("git add .", echo=True, pty=not WINDOWS)
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)
    ctx.run("git push", echo=True, pty=not WINDOWS)


# DVC commands
@task
def dvc_add(ctx: Context, folder: str, message: str) -> None:
    """Add data to DVC and push to remote storage.

    Args:
        folder: Path to the folder or file to add to DVC
        message: Commit message for the changes

    Example:
        invoke dvc-add --folder data/raw --message "Add new training data"
    """
    print(f"Adding {folder} to DVC...")
    ctx.run(f"dvc add {folder}", echo=True, pty=not WINDOWS)

    print("\nStaging DVC files in git...")
    ctx.run(f"git add {folder}.dvc .gitignore", echo=True, pty=not WINDOWS)

    print("\nCommitting changes...")
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)

    print("\nPushing to git remote...")
    ctx.run("git push", echo=True, pty=not WINDOWS)

    print("\nPushing data to DVC remote...")
    ctx.run("dvc push", echo=True, pty=not WINDOWS)

    print(f"\n✓ Successfully added {folder} to DVC and pushed to remotes!")


# W&B Sweep commands
@task
def sweep_init(ctx: Context, config: str = "configs/sweep.yaml", project: str = "mlops_exercises") -> None:
    """Initialize a W&B sweep.

    Args:
        config: Path to sweep configuration YAML file
        project: W&B project name

    Example:
        invoke sweep-init
        invoke sweep-init --config configs/sweep.yaml --project mlops_exercises
    """
    ctx.run(f"wandb sweep {config} --project {project}", echo=True, pty=not WINDOWS)
    print("\n✓ Sweep initialized! Copy the sweep ID and run: invoke sweep-run --sweep-id <ID>")


@task
def sweep_run(
    ctx: Context,
    sweep_id: str,
    entity: str = "",
    project: str = "mlops_exercises",
    count: int = 0,
) -> None:
    """Run a W&B sweep agent.

    Args:
        sweep_id: The sweep ID from sweep-init (e.g., 'abc123xyz')
        entity: W&B entity (username or team). Auto-detected if not provided.
        project: W&B project name
        count: Number of runs to execute (0 = unlimited, uses run_cap from config)

    Example:
        invoke sweep-run --sweep-id abc123xyz
        invoke sweep-run --sweep-id abc123xyz --entity myusername
        invoke sweep-run --sweep-id abc123xyz --count 5
    """
    count_arg = f"--count {count}" if count > 0 else ""

    # Build full sweep path
    if "/" in sweep_id and sweep_id.count("/") >= 2:
        # Already full path: entity/project/sweep_id
        full_sweep_id = sweep_id
    else:
        # Need to get entity
        if not entity:
            # Auto-detect entity from wandb
            import wandb
            api = wandb.Api()
            entity = api.default_entity
            print(f"Using W&B entity: {entity}")

        if "/" in sweep_id:
            # Has project/sweep_id format
            full_sweep_id = f"{entity}/{sweep_id}"
        else:
            # Just sweep_id
            full_sweep_id = f"{entity}/{project}/{sweep_id}"

    ctx.run(f"wandb agent {count_arg} {full_sweep_id}", echo=True, pty=not WINDOWS)


@task
def sweep_link_best(
    ctx: Context,
    sweep_id: str,
    top: int = 3,
    collection: str = "Best-performing-models",
    dry_run: bool = False,
) -> None:
    """Link top performing models from a sweep to the model registry.

    Args:
        sweep_id: The sweep ID (e.g., 'abc123xyz' or 'entity/project/abc123xyz')
        top: Number of top models to link
        collection: Registry collection name
        dry_run: Preview without making changes

    Example:
        invoke sweep-link-best --sweep-id abc123xyz
        invoke sweep-link-best --sweep-id abc123xyz --top 5 --dry-run
        invoke sweep-link-best --sweep-id abc123xyz --collection "Best-performing-models"
    """
    dry_run_flag = "--dry-run" if dry_run else ""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/link_best_models.py {sweep_id} "
        f"--top {top} --collection \"{collection}\" {dry_run_flag}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def model_download(
    ctx: Context,
    alias: str = "best",
    collection: str = "Best-performing-models",
    output: str = "models/registry",
    verify: bool = False,
) -> None:
    """Download a model from the W&B model registry.

    Args:
        alias: Model alias (e.g., 'best', 'top-1', 'latest', 'v0')
        collection: Registry collection name
        output: Directory to download model to
        verify: Run inference on test data to verify the model

    Example:
        invoke model-download
        invoke model-download --alias top-1
        invoke model-download --alias best --verify
        invoke model-download --alias v0 --output models/production
    """
    verify_flag = "--verify" if verify else ""
    ctx.run(
        f"uv run python src/{PROJECT_NAME}/download_model.py "
        f"--alias {alias} --collection \"{collection}\" --output {output} {verify_flag}",
        echo=True,
        pty=not WINDOWS,
    )
