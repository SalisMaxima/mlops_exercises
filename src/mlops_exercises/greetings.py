from pathlib import Path

import typer

app = typer.Typer()


# Hello world command with multiple arguments and options using typer
@app.command()
def greet(name: str = None, count: int = 1) -> None:
    """
    Greet a user with a customizable message.

    Args:
        name: The name of the person to greet. If not provided, greets "World".
        count: The number of times to repeat the greeting. Defaults to 1.
    """
    if name:
        for _ in range(count):
            typer.echo(f"Hello, {name}!")
    else:
        for _ in range(count):
            typer.echo("Hello, World!")


if __name__ == "__main__":
    app()
