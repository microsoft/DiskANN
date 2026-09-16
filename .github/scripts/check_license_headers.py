# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Check or prepend LICENSE_HEADER.txt on all non-ignored Rust files in this repository."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "fix"],
                        help="check is read-only; fix prepends the template without removing content")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="repository root (default: current directory)")
    args = parser.parse_args()
    try:
        template = args.root / ".github" / "LICENSE_HEADER.txt"
        header = template.read_bytes()
        if not header:
            raise ValueError(f"Empty template: {template}")
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.rs"],
            cwd=args.root, capture_output=True, check=True,
        )
        paths = sorted(set(result.stdout.split(b"\0")) - {b""})
        missing = 0
        for name in paths:
            path = args.root / os.fsdecode(name)
            with path.open("rb") as source:
                prefix = source.read(len(header))
            if prefix == header:
                continue
            if args.command == "fix":
                if path.is_symlink() or path.stat().st_nlink != 1:
                    raise ValueError(f"Refusing to write a linked file: {path}")
                path.write_bytes(header + b"\n" + path.read_bytes())
                print(f"Added header: {os.fsdecode(name)!a}")
            else:
                missing += 1
                print(f"Missing template header: {os.fsdecode(name)!a}")
        print(f"Checked {len(paths)} Rust file(s).")
        return int(missing != 0)
    except subprocess.CalledProcessError as error:
        print(f"Git error: {error.stderr.decode(errors='replace')!a}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"Error: {str(error)!a}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
