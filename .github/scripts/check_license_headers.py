# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Check or prepend LICENSE_HEADER.txt on all non-ignored Rust files in this repository."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / ".github" / "LICENSE_HEADER.txt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "fix"],
                        help="check is read-only; fix prepends the template without removing content")
    args = parser.parse_args()
    try:
        header = TEMPLATE.read_bytes()
        if not header:
            raise ValueError(f"Empty template: {TEMPLATE}")
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.rs"],
            cwd=ROOT, capture_output=True, check=True,
        )
        paths = sorted(set(result.stdout.split(b"\0")) - {b""})
        missing = 0
        for name in paths:
            path = ROOT / os.fsdecode(name)
            content = path.read_bytes()
            if content.startswith(header):
                continue
            if args.command == "fix":
                if path.is_symlink() or path.stat().st_nlink != 1:
                    raise ValueError(f"Refusing to write a linked file: {path}")
                path.write_bytes(header + b"\n" + content)
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
