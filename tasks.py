import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_exercises"
PYTHON_VERSION = "3.12"


# Environment commands
@task
def sync(ctx: Context) -> None:
    """Install/sync all dependencies."""
    ctx.run("uv sync", echo=True, pty=not WINDOWS)


@task
def dev(ctx: Context) -> None:
    """Install with dev dependencies."""
    ctx.run("uv sync --dev", echo=True, pty=not WINDOWS)

#Check python path and version
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
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


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


#Git commands
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

