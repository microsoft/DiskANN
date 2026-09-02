# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Checks deterministic repository rules for files changed by a pull request."""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable


BLOCK_HEADER = (
    "/*",
    " * Copyright (c) Microsoft Corporation.",
    " * Licensed under the MIT license.",
    " */",
)
SLASH_HEADER = (
    "// Copyright (c) Microsoft Corporation. All rights reserved.",
    "// Licensed under the MIT license.",
)
HASH_HEADER = (
    "# Copyright (c) Microsoft Corporation. All rights reserved.",
    "# Licensed under the MIT license.",
)

SLASH_HEADER_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".fbs",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".java",
    ".js",
    ".jsx",
    ".pest",
    ".proto",
    ".swift",
    ".ts",
    ".tsx",
}
BLOCK_HEADER_EXTENSIONS = {".rs"}
HASH_HEADER_EXTENSIONS = {
    ".jl",
    ".ps1",
    ".psd1",
    ".psm1",
    ".py",
    ".pyi",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
GENERATED_PATH_PARTS = {
    "__codegen",
    "__generated__",
    "generated",
    "target",
    "third_party",
    "vendor",
}
TEST_INPUT_PATH_PARTS = {"fixtures", "test_data", "test-inputs", "test_inputs"}
TEST_INPUT_PATH_PAIRS = {
    ("tests", "compile-fail"),
    ("tests", "compile_fail"),
}
BINARY_FIXTURE_EXTENSIONS = {
    ".bin",
    ".data",
    ".fbin",
    ".gt10",
    ".ibin",
    ".index",
    ".rangeres",
}
FIXTURE_CONTEXT_PARTS = {
    "benchmarks",
    "benches",
    "example",
    "examples",
    "fixtures",
    "test",
    "test_data",
    "tests",
}
STRUCTURAL_CLIPPY_LINTS = {"single_match", "too_many_arguments"}

ALLOW_RE = re.compile(
    r"^[ \t]*(?P<prefix>#!?)\s*\[\s*allow\s*\((?P<body>.*?)\)\s*\]",
    re.DOTALL | re.MULTILINE,
)
CLIPPY_LINT_RE = re.compile(r"\bclippy::([a-zA-Z0-9_]+)\b")


def normalized_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/"))


def has_adjacent_parts(parts: tuple[str, ...], pair: tuple[str, str]) -> bool:
    return any(parts[index : index + 2] == pair for index in range(len(parts) - 1))


def is_test_input(path: str) -> bool:
    parts = normalized_path(path).parts
    return bool(TEST_INPUT_PATH_PARTS.intersection(parts)) or any(
        has_adjacent_parts(parts, pair) for pair in TEST_INPUT_PATH_PAIRS
    )


def is_generated(path: str, data: bytes) -> bool:
    parts = normalized_path(path).parts
    if GENERATED_PATH_PARTS.intersection(parts):
        return True

    prefix = data[:2048].lower()
    return any(
        marker in prefix
        for marker in (b"@generated", b"automatically generated", b"do not edit")
    )


def expected_headers(path: str) -> tuple[tuple[str, ...], ...]:
    extension = normalized_path(path).suffix.lower()
    if extension in BLOCK_HEADER_EXTENSIONS:
        return (BLOCK_HEADER, SLASH_HEADER)
    if extension in SLASH_HEADER_EXTENSIONS:
        return (SLASH_HEADER,)
    if extension in HASH_HEADER_EXTENSIONS:
        return (HASH_HEADER,)
    return ()


def has_expected_header(data: bytes, header: tuple[str, ...]) -> bool:
    try:
        lines = data.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return False

    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    return tuple(lines[: len(header)]) == header


def check_source_header(path: str, data: bytes) -> list[str]:
    headers = expected_headers(path)
    if not headers or is_test_input(path) or is_generated(path, data):
        return []
    if any(has_expected_header(data, header) for header in headers):
        return []
    return [f"{path}: newly added source file is missing the standard license header"]


def looks_binary(data: bytes) -> bool:
    if b"\0" in data:
        return True
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    if not text:
        return False
    control_characters = sum(
        character < " " and character not in "\n\r\t" for character in text
    )
    return control_characters / len(text) > 0.01


def is_binary_fixture(path: str, data: bytes) -> bool:
    parsed = normalized_path(path)
    if parsed.suffix.lower() in BINARY_FIXTURE_EXTENSIONS:
        return True
    return bool(FIXTURE_CONTEXT_PARTS.intersection(parsed.parts)) and looks_binary(data)


def check_binary_fixture(
    path: str,
    data: bytes,
    lfs_filter: Callable[[str], str | None],
) -> list[str]:
    if not is_binary_fixture(path, data):
        return []

    violations = []
    if "test_data" not in normalized_path(path).parts:
        violations.append(
            f"{path}: binary fixture must be stored under a test_data directory"
        )
    if lfs_filter(path) != "lfs":
        violations.append(f"{path}: binary fixture must be tracked by Git LFS")
    return violations


def allow_annotations(source: str) -> list[tuple[bool, frozenset[str], bool, int]]:
    annotations = []
    for match in ALLOW_RE.finditer(source):
        lints = frozenset(CLIPPY_LINT_RE.findall(match.group("body")))
        targeted_lints = lints.intersection(STRUCTURAL_CLIPPY_LINTS)
        if targeted_lints:
            annotations.append(
                (
                    match.group("prefix") == "#!",
                    frozenset(targeted_lints),
                    bool(re.search(r"\breason\s*=", match.group("body"))),
                    source.count("\n", 0, match.start()) + 1,
                )
            )
    return annotations


def check_new_structural_allows(path: str, before: str, after: str) -> list[str]:
    before_counts = collections.Counter(
        annotation[:3] for annotation in allow_annotations(before)
    )
    violations = []

    for inner, lints, has_reason, line in allow_annotations(after):
        key = (inner, lints, has_reason)
        if before_counts[key]:
            before_counts[key] -= 1
            continue

        lint_names = ", ".join(f"clippy::{lint}" for lint in sorted(lints))
        if inner:
            violations.append(
                f"{path}:{line}: new crate-wide allow for {lint_names} is not permitted"
            )
        elif not has_reason:
            violations.append(
                f"{path}:{line}: new allow for {lint_names} must include reason = \"...\""
            )

    return violations


def run_git(repo: Path, *arguments: str, check: bool = True) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def changed_files(repo: Path, base: str) -> list[tuple[str, str, str]]:
    output = run_git(
        repo,
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "--find-copies",
        "--diff-filter=ACMR",
        f"{base}...HEAD",
    )
    fields = iter(field.decode("utf-8") for field in output.split(b"\0") if field)
    changes = []
    for status in fields:
        if status.startswith(("C", "R")):
            old_path = next(fields)
            new_path = next(fields)
        else:
            old_path = new_path = next(fields)
        changes.append((status, old_path, new_path))
    return changes


def file_at_revision(repo: Path, revision: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode:
        return ""
    return result.stdout.decode("utf-8", errors="replace")


def git_lfs_filter(repo: Path, path: str) -> str | None:
    output = run_git(repo, "check-attr", "filter", "--", path)
    value = output.decode("utf-8", errors="replace").rsplit(": ", 1)[-1].strip()
    return None if value in {"set", "unspecified", "unset"} else value


def collect_violations(repo: Path, base_ref: str) -> list[str]:
    merge_base = run_git(repo, "merge-base", base_ref, "HEAD").decode().strip()
    violations = []

    for status, old_path, path in changed_files(repo, merge_base):
        full_path = repo / Path(path)
        if not full_path.is_file():
            continue
        data = full_path.read_bytes()

        if status.startswith(("A", "C")):
            violations.extend(check_source_header(path, data))
            violations.extend(
                check_binary_fixture(
                    path,
                    data,
                    lambda candidate: git_lfs_filter(repo, candidate),
                )
            )

        if normalized_path(path).suffix.lower() == ".rs" and not is_generated(path, data):
            before = file_at_revision(repo, merge_base, old_path)
            after = data.decode("utf-8", errors="replace")
            violations.extend(check_new_structural_allows(path, before, after))

    return violations


def collect_worktree_violations(repo: Path) -> list[str]:
    status = run_git(repo, "status", "--porcelain", "--untracked-files=all")
    return [
        f"test suite left the worktree dirty: {line}"
        for line in status.decode("utf-8", errors="replace").splitlines()
    ]


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base", help="Git revision used as the PR base")
    mode.add_argument(
        "--worktree-clean",
        action="store_true",
        help="Fail when tracked or non-ignored untracked files are dirty",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(arguments)
    try:
        repo = args.repo.resolve()
        violations = (
            collect_worktree_violations(repo)
            if args.worktree_clean
            else collect_violations(repo, args.base)
        )
    except (OSError, RuntimeError) as error:
        print(f"error: unable to check deterministic rules: {error}", file=sys.stderr)
        return 2

    if not violations:
        print("Deterministic repository rules passed.")
        return 0

    print("Deterministic repository rule violations:", file=sys.stderr)
    for violation in violations:
        print(f"  - {violation}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
