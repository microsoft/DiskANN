# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Reject multiple versions used directly by workspace members.

Checks the first hop of Cargo's resolved graph with all features and no platform
filter, including normal, dev, and build dependencies. Transitive-only versions
are ignored. Different requirements resolving to one version are allowed.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def find_duplicates(metadata: dict) -> list[str]:
    packages = {package["id"]: package for package in metadata["packages"]}
    members = set(metadata["workspace_members"])
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    if not members or not members <= nodes.keys():
        raise ValueError("Expected a resolved dependency graph for every workspace member")

    direct = {}
    for member in sorted(members):
        consumer = packages[member]
        for dependency in nodes[member]["deps"]:
            package = packages[dependency["pkg"]]
            versions = direct.setdefault(package["name"], {})
            consumers = versions.setdefault(package["version"], set())
            consumers.add(f"{consumer['name']} ({consumer['manifest_path']})")

    errors = []
    for name, versions in sorted(direct.items()):
        if len(versions) > 1:
            details = "\n".join(
                f"  {version}: {', '.join(sorted(consumers))}"
                for version, consumers in sorted(versions.items())
            )
            errors.append(f"{name} has multiple directly used versions:\n{details}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("Cargo.toml"),
        help="workspace manifest (default: Cargo.toml in the current directory)",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest_path.resolve()
    command = [
        "cargo", "metadata", "--locked", "--format-version", "1", "--all-features",
        "--manifest-path", str(manifest),
    ]
    try:
        result = subprocess.run(
            command, cwd=manifest.parent, check=True, capture_output=True,
            text=True, encoding="utf-8",
        )
        errors = find_duplicates(json.loads(result.stdout))
    except subprocess.CalledProcessError as error:
        print(f"cargo metadata failed (exit {error.returncode}):\n{error.stderr}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Direct dependency version check failed: {error}", file=sys.stderr)
        return 2

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Direct dependency versions agree (all features, all platforms, normal/dev/build).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
