# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

from check_pr_rules import (
    BLOCK_HEADER,
    HASH_HEADER,
    SLASH_HEADER,
    check_binary_fixture,
    check_new_structural_allows,
    check_source_header,
    collect_violations,
    collect_worktree_violations,
    main,
)


class SourceHeaderTests(unittest.TestCase):
    def test_accepts_block_header(self) -> None:
        source = "\n".join((*BLOCK_HEADER, "", "fn main() {}")).encode()
        self.assertEqual(check_source_header("src/main.rs", source), [])

    def test_accepts_rust_line_header(self) -> None:
        source = "\n".join((*SLASH_HEADER, "", "fn main() {}")).encode()
        self.assertEqual(check_source_header("src/main.rs", source), [])

    def test_accepts_slash_header_for_schema(self) -> None:
        source = "\n".join((*SLASH_HEADER, "", 'syntax = "proto3";')).encode()
        self.assertEqual(check_source_header("src/schema.proto", source), [])

    def test_accepts_hash_header_after_shebang(self) -> None:
        source = "\n".join(("#!/bin/sh", *HASH_HEADER, "", "set -eu")).encode()
        self.assertEqual(check_source_header("tools/check.sh", source), [])

    def test_rejects_missing_header(self) -> None:
        self.assertEqual(
            len(check_source_header("src/main.rs", b"fn main() {}\n")),
            1,
        )

    def test_ignores_markdown_and_test_inputs(self) -> None:
        self.assertEqual(check_source_header("README.md", b"text\n"), [])
        self.assertEqual(
            check_source_header("tests/fixtures/input.rs", b"invalid rust\n"),
            [],
        )
        self.assertEqual(
            check_source_header("tests/compile-fail/input.rs", b"invalid rust\n"),
            [],
        )

    def test_ignores_generated_sources(self) -> None:
        self.assertEqual(
            check_source_header("src/generated/bindings.rs", b"pub struct Binding;\n"),
            [],
        )
        self.assertEqual(
            check_source_header("src/bindings.rs", b"// @generated\npub struct Binding;\n"),
            [],
        )


class BinaryFixtureTests(unittest.TestCase):
    def test_accepts_nested_lfs_fixture(self) -> None:
        self.assertEqual(
            check_binary_fixture(
                "crate/test_data/input.bin",
                b"\0binary",
                lambda _: "lfs",
            ),
            [],
        )

    def test_rejects_fixture_outside_test_data(self) -> None:
        violations = check_binary_fixture(
            "crate/tests/input.bin",
            b"\0binary",
            lambda _: "lfs",
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("test_data", violations[0])

    def test_rejects_fixture_without_lfs(self) -> None:
        violations = check_binary_fixture(
            "test_data/input.fbin",
            b"\0binary",
            lambda _: None,
        )
        self.assertEqual(len(violations), 1)
        self.assertIn("Git LFS", violations[0])

    def test_ignores_text_examples(self) -> None:
        self.assertEqual(
            check_binary_fixture(
                "crate/examples/input.json",
                b'{"value": 1}\n',
                lambda _: None,
            ),
            [],
        )

    def test_detects_unknown_binary_extension_in_test_context(self) -> None:
        violations = check_binary_fixture(
            "crate/tests/input.fixture",
            b"\xff\x00",
            lambda _: None,
        )
        self.assertEqual(len(violations), 2)


class StructuralAllowTests(unittest.TestCase):
    def test_accepts_existing_allow(self) -> None:
        source = "#[allow(clippy::too_many_arguments)]\nfn f() {}\n"
        self.assertEqual(check_new_structural_allows("src/lib.rs", source, source), [])

    def test_accepts_new_allow_with_reason(self) -> None:
        after = (
            '#[allow(clippy::too_many_arguments, reason = "external ABI")]\n'
            "fn f() {}\n"
        )
        self.assertEqual(check_new_structural_allows("src/lib.rs", "", after), [])

    def test_rejects_new_allow_without_reason(self) -> None:
        after = "#[allow(clippy::single_match)]\nfn f() {}\n"
        violations = check_new_structural_allows("src/lib.rs", "", after)
        self.assertEqual(len(violations), 1)
        self.assertIn('reason = "..."', violations[0])

    def test_rejects_new_crate_wide_allow(self) -> None:
        after = '#![allow(clippy::too_many_arguments, reason = "legacy")]\n'
        violations = check_new_structural_allows("src/lib.rs", "", after)
        self.assertEqual(len(violations), 1)
        self.assertIn("crate-wide", violations[0])

    def test_handles_multiline_allow(self) -> None:
        after = """#[allow(
    clippy::too_many_arguments,
    reason = "trait signature"
)]
fn f() {}
"""
        self.assertEqual(check_new_structural_allows("src/lib.rs", "", after), [])


class GitIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        self.git("init", "--initial-branch=main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test User")
        (self.repo / ".gitattributes").write_text(
            "**/test_data/**/*.bin filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
        (self.repo / ".gitignore").write_text("target/\n", encoding="utf-8")
        (self.repo / "README.md").write_text("baseline\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "baseline")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def git(self, *arguments: str) -> None:
        subprocess.run(
            ["git", *arguments],
            cwd=self.repo,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_accepts_valid_added_files(self) -> None:
        source = "\n".join((*HASH_HEADER, "", "print('ok')", "")).encode()
        (self.repo / "check.py").write_bytes(source)
        fixture = self.repo / "crate" / "test_data" / "input.bin"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"\0binary")
        self.git("add", ".")
        self.git("commit", "-m", "valid changes")

        self.assertEqual(collect_violations(self.repo, "HEAD^"), [])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                main(["--base", "HEAD^", "--repo", str(self.repo)]),
                0,
            )

    def test_reports_representative_bad_inputs(self) -> None:
        (self.repo / "bad.py").write_text("print('missing header')\n", encoding="utf-8")
        fixture = self.repo / "crate" / "tests" / "input.bin"
        fixture.parent.mkdir(parents=True)
        fixture.write_bytes(b"\0binary")
        source = self.repo / "crate" / "src" / "lib.rs"
        source.parent.mkdir(parents=True)
        source.write_text(
            "\n".join(
                (
                    *BLOCK_HEADER,
                    "",
                    "#[allow(clippy::too_many_arguments)]",
                    "fn f() {}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-m", "invalid changes")

        violations = collect_violations(self.repo, "HEAD^")

        self.assertEqual(len(violations), 4)
        self.assertTrue(any("license header" in violation for violation in violations))
        self.assertTrue(any("test_data" in violation for violation in violations))
        self.assertTrue(any("Git LFS" in violation for violation in violations))
        self.assertTrue(any('reason = "..."' in violation for violation in violations))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                main(["--base", "HEAD^", "--repo", str(self.repo)]),
                1,
            )

    def test_rename_preserves_existing_structural_allow(self) -> None:
        source = self.repo / "src" / "old.rs"
        source.parent.mkdir()
        source.write_text(
            "\n".join(
                (
                    *BLOCK_HEADER,
                    "",
                    "#[allow(clippy::too_many_arguments)]",
                    "fn f() {}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-m", "add legacy source")
        self.git("mv", "src/old.rs", "src/new.rs")
        self.git("commit", "-m", "rename source")

        self.assertEqual(collect_violations(self.repo, "HEAD^"), [])

    def test_worktree_check_ignores_ignored_build_outputs(self) -> None:
        output = self.repo / "target" / "output"
        output.parent.mkdir()
        output.write_text("build output\n", encoding="utf-8")

        self.assertEqual(collect_worktree_violations(self.repo), [])

    def test_worktree_check_reports_tracked_and_untracked_changes(self) -> None:
        (self.repo / "README.md").write_text("modified\n", encoding="utf-8")
        (self.repo / "unexpected.txt").write_text("new\n", encoding="utf-8")

        violations = collect_worktree_violations(self.repo)

        self.assertEqual(len(violations), 2)
        self.assertTrue(any("README.md" in violation for violation in violations))
        self.assertTrue(any("unexpected.txt" in violation for violation in violations))


if __name__ == "__main__":
    unittest.main()
