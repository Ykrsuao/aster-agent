# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Choose which folders may configure and extend Aster with their own files.

Usage:
    aster trust
    aster trust path/to/repository
    aster trust --list
    aster trust --remove path/to/repository
"""

from pathlib import Path

import click


@click.command()
@click.argument(
    "folder",
    required=False,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--remove", is_flag=True, help="Remove the folder from the trusted list.")
@click.option("--list", "show", is_flag=True, help="Show the trusted folders.")
def command(folder: Path | None, remove: bool, show: bool) -> None:
    """Trust a folder so Aster uses the skills, libraries and settings inside it.

    Aster imports Python from a trusted folder and reads its model settings and
    secrets, so trust only folders whose contents you would run yourself.
    FOLDER defaults to the current directory; folders inside it are trusted too.

    \b
    Examples:
        aster trust
        aster trust --list
        aster trust --remove .
    """
    from aster import workspace_trust

    if show:
        if folder is not None or remove:
            raise click.UsageError("Use --list on its own.")
        entries = workspace_trust.trusted_workspaces()
        if not entries:
            click.echo("No folders are trusted.")
        for entry in entries:
            click.echo(entry)
        return

    target = (folder or Path.cwd()).resolve()
    try:
        if remove:
            removed = workspace_trust.untrust(target)
            click.echo(
                f"Removed trust entry: {target}" if removed else f"Not in the list: {target}"
            )
        else:
            workspace_trust.trust(target)
            click.echo(f"Trusted: {target}")
    except OSError as exc:
        raise click.ClickException(f"Could not save the trusted folders: {exc}") from exc
