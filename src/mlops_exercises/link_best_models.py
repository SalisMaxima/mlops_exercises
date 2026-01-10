"""
Link top performing models from a W&B sweep to the model registry.

This script finds the best models from a sweep based on test_accuracy
and registers them to the W&B Model Registry with appropriate aliases.

Usage:
    python link_best_models.py <sweep_id> --top 3 --collection "Best-performing-models"
    python link_best_models.py abc123xyz --dry-run
"""

import typer
import wandb
from loguru import logger

app = typer.Typer()


def get_sweep_runs(api: wandb.Api, sweep_path: str) -> list:
    """Get all runs from a sweep, sorted by test_accuracy descending."""
    sweep = api.sweep(sweep_path)
    runs = list(sweep.runs)

    # Filter runs that have test_accuracy and sort by it
    runs_with_accuracy = []
    for run in runs:
        test_accuracy = run.summary.get("test_accuracy")
        if test_accuracy is not None:
            runs_with_accuracy.append((run, test_accuracy))

    # Sort by test_accuracy descending
    runs_with_accuracy.sort(key=lambda x: x[1], reverse=True)

    return runs_with_accuracy


def get_model_artifact_from_run(run: wandb.apis.public.Run) -> wandb.Artifact | None:
    """Get the model artifact from a run."""
    for artifact in run.logged_artifacts():
        if artifact.type == "model":
            return artifact
    return None


@app.command()
def link_best_models(
    sweep_id: str = typer.Argument(..., help="Sweep ID (e.g., 'abc123xyz' or 'entity/project/abc123xyz')"),
    top: int = typer.Option(3, "--top", "-n", help="Number of top models to link"),
    collection: str = typer.Option("Best-performing-models", "--collection", "-c", help="Registry collection name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without making changes"),
) -> None:
    """Link top performing models from a sweep to the model registry."""
    api = wandb.Api()

    # Build full sweep path if not provided
    if "/" not in sweep_id:
        entity = api.default_entity
        project = "mlops_exercises"
        sweep_path = f"{entity}/{project}/{sweep_id}"
    elif sweep_id.count("/") == 1:
        # project/sweep_id format
        entity = api.default_entity
        sweep_path = f"{entity}/{sweep_id}"
    else:
        # Full path provided
        sweep_path = sweep_id

    logger.info(f"Fetching runs from sweep: {sweep_path}")

    # Get runs sorted by test_accuracy
    runs_with_accuracy = get_sweep_runs(api, sweep_path)

    if not runs_with_accuracy:
        logger.error("No runs with test_accuracy found in this sweep")
        raise typer.Exit(1)

    logger.info(f"Found {len(runs_with_accuracy)} runs with test_accuracy")

    # Get entity for registry path
    entity = sweep_path.split("/")[0]

    # Link top N models
    linked_count = 0
    for rank, (run, test_accuracy) in enumerate(runs_with_accuracy[:top], start=1):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Rank {rank}: Run {run.name} (test_accuracy={test_accuracy:.4f})")

        artifact = get_model_artifact_from_run(run)
        if artifact is None:
            logger.warning(f"  No model artifact found for run {run.name}, skipping")
            continue

        logger.info(f"  Artifact: {artifact.name}:{artifact.version}")

        # Build aliases
        aliases = [f"top-{rank}"]
        if rank == 1:
            aliases.append("best")

        # Build registry path
        registry_path = f"{entity}/model-registry/{collection}"

        if dry_run:
            logger.info(f"  [DRY RUN] Would link to: {registry_path}")
            logger.info(f"  [DRY RUN] With aliases: {aliases}")
        else:
            try:
                artifact.link(target_path=registry_path, aliases=aliases)
                artifact.save()
                logger.info(f"  Linked to: {registry_path}")
                logger.info(f"  Aliases: {aliases}")
                linked_count += 1
            except Exception as e:
                logger.error(f"  Failed to link: {e}")

    logger.info(f"\n{'=' * 50}")
    if dry_run:
        logger.info(f"[DRY RUN] Would have linked {min(top, len(runs_with_accuracy))} models")
    else:
        logger.info(f"Successfully linked {linked_count} models to {collection}")


if __name__ == "__main__":
    app()
