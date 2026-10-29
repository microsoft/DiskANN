# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""Test the pre-checkout storage steps without accessing or changing disks."""

from pathlib import Path
import shlex
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ("disk-benchmarks.yml", "disk-benchmarks-aa.yml")


def storage_step(workflow):
    content = (ROOT / ".github" / "workflows" / workflow).read_text()
    step = content.split("- name: Use mounted high-speed NVMe SSD\n", 1)[1]
    return textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - ", 1)[0])


MOCKS = r"""
lsblk() { :; }
findmnt() {
  if [[ "$1" == --real ]]; then
    printf '%s\n' "$MOUNTS"
  elif [[ "$1" == --mountpoint ]]; then
    printf '%s\n' "$RESOLVED_SOURCE"
  else
    return 1
  fi
}
sudo() {
  case "$1" in
    mkdir|chown) printf 'WRITE %s\n' "$*" ;;
    *) echo "Unexpected privileged command: $*" >&2; return 1 ;;
  esac
}
ln() { printf 'LINK %s\n' "$*"; }
"""


class StorageSetupTests(unittest.TestCase):
    def run_step(self, mounts, source="", success=False):
        fixtures = f"MOUNTS={shlex.quote(mounts)}\nRESOLVED_SOURCE={shlex.quote(source)}\n"
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail"],
            input=(fixtures + MOCKS + storage_step(WORKFLOWS[0])).encode(),
            capture_output=True,
            check=False,
        )
        result.stdout = result.stdout.decode()
        result.stderr = result.stderr.decode()
        self.assertEqual(result.stderr, "")
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("::error::", result.stdout)
            self.assertNotIn("WRITE ", result.stdout)
            self.assertNotIn("LINK ", result.stdout)
        return result.stdout

    def test_workflows_use_identical_storage_steps(self):
        self.assertEqual(storage_step(WORKFLOWS[0]), storage_step(WORKFLOWS[1]))

    def test_mounted_nvme(self):
        for source, target in (
            ("/dev/nvme0n1p1", "/mnt/storage"),
            ("/dev/nvme0n1", "/mnt/nvme"),
            ("/dev/nvme0n1p2", "/mnt/custom-nvme"),
        ):
            with self.subTest(source=source, target=target):
                output = self.run_step(
                    f"/dev/sda1 /\n/dev/sdb1 /mnt\n{source} {target}",
                    source, success=True,
                )
                self.assertIn(f"LINK -s {target}/diskann_rust diskann_rust", output)
                self.assertIn(f"LINK -s {target}/baseline baseline", output)
                self.assertIn(f"WRITE mkdir -p {target}/diskann_rust {target}/baseline", output)
                self.assertIn("WRITE chown ", output)

    def test_missing_or_wrong_storage(self):
        for mounts in (
            "",  # No mounted NVMe, including an uninitialized bare disk.
            "/dev/sda1 /",  # An ordinary directory on the OS filesystem.
            "/dev/sdb1 /mnt/storage",  # Resource disk at the expected-looking path.
            "/dev/nvme1n1p1 /mnt/storage",  # Not the target NVMe.
            "/dev/nvme0n1p1 /",
            "/dev/nvme0n1p1 /boot/efi",
            "/dev/nvme0n1p1 /\n/dev/nvme0n1p2 /mnt/storage",
            "/dev/nvme0n1p1 /mnt/one\n/dev/nvme0n1p2 /mnt/two",
        ):
            with self.subTest(mounts=mounts):
                self.run_step(mounts)

    def test_mount_path_must_resolve_to_nvme(self):
        for source in ("", "/dev/sda1", "/dev/sdb1", "/dev/nvme1n1p1"):
            with self.subTest(source=source):
                self.run_step("/dev/nvme0n1p1 /mnt/storage", source)


if __name__ == "__main__":
    unittest.main()
