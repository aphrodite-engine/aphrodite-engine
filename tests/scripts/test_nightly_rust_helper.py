# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("script", ["build_nightly_commits.sh", "reconcile_platform_wheels.sh"])
@pytest.mark.parametrize("build_status", [0, 1])
def test_backlog_uses_current_helper_and_restores_checkout(tmp_path, script, build_status):
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/scripts" / script).read_text()
    function = (
        "build_with_current_rust_helper() ("
        + source.split("build_with_current_rust_helper() (", 1)[1].split("\n)", 1)[0]
        + "\n)"
    )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "tools").mkdir()
    helper = tmp_path / "tools/build_rust.py"
    helper.write_text("old helper\n")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "old")
    old = git("rev-parse", "HEAD")
    helper.write_text("new helper\n")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "new")
    new = git("rev-parse", "HEAD")
    git("checkout", "--detach", old)
    (tmp_path / "docker").mkdir()
    build = tmp_path / "docker/export_wheels.sh"
    build.write_text(
        "#!/bin/bash\nset -eu\n"
        'test "$(<tools/build_rust.py)" = "new helper"\n'
        'test "$APHRODITE_VERSION_OVERRIDE" = "0.24.0"\n'
        f"exit {build_status}\n"
    )
    build.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            function + '\ntarget_commit="$1"\n'
            "PLATFORM_BUILD_SCRIPT=./docker/export_wheels.sh\noutput_dir=wheels\n"
            'build_with_current_rust_helper "$2" 0.24.0',
            "test",
            new,
            old,
        ],
        cwd=tmp_path,
    )
    assert result.returncode == build_status
    assert helper.read_text() == "old helper\n"
    assert git("diff", "--name-only") == ""
