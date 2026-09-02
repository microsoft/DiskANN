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

GENERATED_PREAMBLE_RE = re.compile(
    r"""(?ix)^\s*(?://+|\#+|\*+|/\*+)\s*
    (?:
        (?:this\s+file\s+is\s+)?@generated\b
        | this\s+file\s+(?:is|was)\s+automatically\s+generated\b
        | automatically\s+generated\s+by\b
        | code\s+(?:is|was)\s+automatically\s+generated\b
        | code\s+generated\b.*\bdo\s+not\s+edit\b
        | do\s+not\s+edit\b.*\bgenerated\b
    )"""
)
LFS_POINTER_RE = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\r?\n"
    rb"oid sha256:[0-9a-f]{64}\r?\n"
    rb"size [0-9]+\r?\n?\Z"
)


class RustLexError(ValueError):
    def __init__(self, source: str, offset: int, message: str) -> None:
        self.line = source.count("\n", 0, offset) + 1
        super().__init__(message)


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

    try:
        lines = data.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return False

    offset = 1 if lines and lines[0].startswith("#!") else 0
    for header in (BLOCK_HEADER, SLASH_HEADER, HASH_HEADER):
        if tuple(lines[offset : offset + len(header)]) == header:
            offset += len(header)
            break

    candidates = []
    for line in lines[offset:]:
        if not line.strip():
            continue
        candidates.append(line)
        if len(candidates) == 4:
            break
    return any(GENERATED_PREAMBLE_RE.match(line) for line in candidates)


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


def is_lfs_pointer(data: bytes) -> bool:
    return bool(LFS_POINTER_RE.fullmatch(data))


def is_binary_fixture(path: str, data: bytes, lfs_filter: str | None = None) -> bool:
    parsed = normalized_path(path)
    if parsed.suffix.lower() in BINARY_FIXTURE_EXTENSIONS:
        return True
    return bool(FIXTURE_CONTEXT_PARTS.intersection(parsed.parts)) and (
        looks_binary(data)
        or lfs_filter == "lfs"
        or is_lfs_pointer(data)
    )


def check_binary_fixture(
    path: str,
    data: bytes,
    lfs_filter: Callable[[str], str | None],
) -> list[str]:
    filter_value = lfs_filter(path)
    if not is_binary_fixture(path, data, filter_value):
        return []

    violations = []
    if "test_data" not in normalized_path(path).parts:
        violations.append(
            f"{path}: binary fixture must be stored under a test_data directory"
        )
    if filter_value != "lfs":
        violations.append(f"{path}: binary fixture must be tracked by Git LFS")
    return violations


def skip_block_comment(source: str, offset: int) -> int:
    depth = 1
    index = offset + 2
    while index < len(source):
        if source.startswith("/*", index):
            depth += 1
            index += 2
        elif source.startswith("*/", index):
            depth -= 1
            index += 2
            if depth == 0:
                return index
        else:
            index += 1
    raise RustLexError(source, offset, "unterminated block comment")


def skip_quoted_literal(source: str, quote: int, delimiter: str) -> int:
    index = quote + 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
        elif source[index] == delimiter:
            return index + 1
        else:
            index += 1
    raise RustLexError(source, quote, f"unterminated {delimiter} literal")


def skip_raw_string(source: str, offset: int) -> int | None:
    index = offset
    if source.startswith(("br", "cr"), index):
        index += 2
    elif source.startswith("r", index):
        index += 1
    else:
        return None

    hash_start = index
    while index < len(source) and source[index] == "#":
        index += 1
    if index >= len(source) or source[index] != '"':
        return None

    terminator = '"' + "#" * (index - hash_start)
    end = source.find(terminator, index + 1)
    if end < 0:
        raise RustLexError(source, offset, "unterminated raw string literal")
    return end + len(terminator)


def skip_char_literal(source: str, quote: int) -> int | None:
    if quote + 1 >= len(source):
        return None
    if source[quote + 1] != "\\":
        return quote + 3 if source[quote + 2 : quote + 3] == "'" else None
    return skip_quoted_literal(source, quote, "'")


def rust_non_code(source: str, offset: int) -> tuple[str, int] | None:
    if source.startswith("//", offset):
        end = source.find("\n", offset + 2)
        return ("comment", len(source) if end < 0 else end)
    if source.startswith("/*", offset):
        return ("comment", skip_block_comment(source, offset))

    raw_end = skip_raw_string(source, offset)
    if raw_end is not None:
        return ("literal", raw_end)

    if source.startswith(('b"', 'c"'), offset):
        return ("literal", skip_quoted_literal(source, offset + 1, '"'))
    if source[offset] == '"':
        return ("literal", skip_quoted_literal(source, offset, '"'))
    if source.startswith("b'", offset):
        end = skip_char_literal(source, offset + 1)
        return ("literal", end) if end is not None else None
    if source[offset] == "'":
        end = skip_char_literal(source, offset)
        return ("literal", end) if end is not None else None
    return None


def rust_attributes(source: str) -> list[tuple[bool, str, int, int]]:
    attributes = []
    index = 0
    delimiter_pairs = {")": "(", "]": "[", "}": "{"}

    while index < len(source):
        non_code = rust_non_code(source, index)
        if non_code is not None:
            index = non_code[1]
            continue
        if source[index] != "#":
            index += 1
            continue

        start = index
        index += 1
        inner = index < len(source) and source[index] == "!"
        if inner:
            index += 1
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source) or source[index] != "[":
            continue

        body_start = index + 1
        stack = ["["]
        index += 1
        while index < len(source) and stack:
            non_code = rust_non_code(source, index)
            if non_code is not None:
                index = non_code[1]
                continue
            character = source[index]
            if character in "([{":
                stack.append(character)
            elif character in ")]}":
                if delimiter_pairs[character] != stack[-1]:
                    raise RustLexError(
                        source, index, "mismatched delimiter in attribute"
                    )
                stack.pop()
                if not stack:
                    attributes.append(
                        (
                            inner,
                            source[body_start:index],
                            source.count("\n", 0, start) + 1,
                            body_start,
                        )
                    )
            index += 1
        if stack:
            raise RustLexError(source, start, "unterminated attribute")

    return attributes


def rust_tokens(source: str) -> list[str]:
    tokens = []
    index = 0
    while index < len(source):
        if source[index].isspace():
            index += 1
            continue
        non_code = rust_non_code(source, index)
        if non_code is not None:
            if non_code[0] == "literal":
                tokens.append("<literal>")
            index = non_code[1]
            continue
        if source[index].isalpha() or source[index] == "_":
            end = index + 1
            while end < len(source) and (
                source[end].isalnum() or source[end] == "_"
            ):
                end += 1
            tokens.append(source[index:end])
            index = end
        elif source.startswith("::", index):
            tokens.append("::")
            index += 2
        else:
            tokens.append(source[index])
            index += 1
    return tokens


def closing_delimiter(tokens: list[str], opening: int, end: int) -> int:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = [tokens[opening]]
    for index in range(opening + 1, end):
        token = tokens[index]
        if token in pairs:
            stack.append(token)
        elif token in pairs.values():
            if not stack or token != pairs[stack[-1]]:
                raise ValueError("mismatched delimiter")
            stack.pop()
            if not stack:
                return index
    raise ValueError("unterminated delimiter")


def split_meta_arguments(tokens: list[str], start: int, end: int) -> list[list[str]]:
    arguments = []
    argument_start = start
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index in range(start, end):
        token = tokens[index]
        if token in pairs:
            stack.append(token)
        elif token in pairs.values():
            if not stack or token != pairs[stack[-1]]:
                raise ValueError("mismatched delimiter")
            stack.pop()
        elif token == "," and not stack:
            arguments.append(tokens[argument_start:index])
            argument_start = index + 1
    if stack:
        raise ValueError("unterminated delimiter")
    arguments.append(tokens[argument_start:end])
    return arguments


def allow_metas(tokens: list[str]) -> list[tuple[frozenset[str], bool]]:
    if not tokens:
        return []
    name = tokens[0]
    if name not in {"allow", "cfg_attr"}:
        return []
    if len(tokens) < 3 or tokens[1] != "(":
        raise ValueError(f"malformed {name} attribute")
    close = closing_delimiter(tokens, 1, len(tokens))
    if close != len(tokens) - 1:
        raise ValueError(f"unexpected tokens after {name} attribute")

    arguments = split_meta_arguments(tokens, 2, close)
    if name == "cfg_attr":
        if len(arguments) < 2:
            raise ValueError("cfg_attr must contain a condition and an attribute")
        output = []
        for argument in arguments[1:]:
            output.extend(allow_metas(argument))
        return output

    lints = set()
    has_reason = False
    for argument in arguments:
        if len(argument) >= 3 and argument[:2] == ["reason", "="]:
            has_reason = True
        if (
            len(argument) == 3
            and argument[0] == "clippy"
            and argument[1] == "::"
            and argument[2] in STRUCTURAL_CLIPPY_LINTS
        ):
            lints.add(argument[2])
    return [(frozenset(lints), has_reason)] if lints else []


def allow_annotations(source: str) -> list[tuple[bool, frozenset[str], bool, int]]:
    annotations = []
    for inner, body, line, body_offset in rust_attributes(source):
        try:
            metas = allow_metas(rust_tokens(body))
        except (RustLexError, ValueError) as error:
            raise RustLexError(source, body_offset, str(error)) from error
        annotations.extend((inner, lints, has_reason, line) for lints, has_reason in metas)
    return annotations


def check_new_structural_allows(path: str, before: str, after: str) -> list[str]:
    try:
        before_counts = collections.Counter(
            annotation[:3] for annotation in allow_annotations(before)
        )
        after_annotations = allow_annotations(after)
    except RustLexError as error:
        return [
            f"{path}:{error.line}: malformed Rust attribute prevents structural lint validation"
        ]
    violations = []

    for inner, lints, has_reason, line in after_annotations:
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
        "--find-copies-harder",
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

        if status.startswith(("A", "C", "R")):
            violations.extend(
                check_binary_fixture(
                    path,
                    data,
                    lambda candidate: git_lfs_filter(repo, candidate),
                )
            )

        if normalized_path(path).suffix.lower() == ".rs" and not is_generated(path, data):
            before = (
                file_at_revision(repo, merge_base, old_path)
                if status.startswith(("M", "R"))
                else ""
            )
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
