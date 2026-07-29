# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest
from packaging.tags import Tag

from aphrodite.build_utils import precompiled

WHEEL_NAME = "aphrodite_engine-0.0.0-cp38-abi3-linux_x86_64.whl"


def _make_wheel(path: Path, files: dict[str, bytes]) -> precompiled.WheelCandidate:
    with zipfile.ZipFile(path, "w") as wheel:
        for filename, contents in files.items():
            wheel.writestr(filename, contents)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return precompiled.WheelCandidate(str(path), path.name, digest)


def test_compatible_candidate_ignores_non_wheel_links(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    monkeypatch.setattr(
        precompiled,
        "sys_tags",
        lambda: iter([Tag("cp38", "abi3", "linux_x86_64")]),
    )
    html = f"""
    <a href="/cdn-cgi/content?id=canary" style="display: none"></a>
    <a href="../{WHEEL_NAME}#sha256={digest}">
      <span class="filename">{WHEEL_NAME}</span>
      <span class="icon">↓</span>
    </a>
    """

    candidate = precompiled._compatible_candidate("https://example.test/commit/aphrodite-engine/index.html", html)

    assert candidate.filename == WHEEL_NAME
    assert candidate.sha256 == digest
    assert candidate.url_or_path == f"https://example.test/commit/{WHEEL_NAME}#sha256={digest}"


def test_explicit_location_takes_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheel = tmp_path / WHEEL_NAME
    wheel.touch()
    monkeypatch.setattr(
        precompiled,
        "get_base_commit_in_main",
        lambda _: pytest.fail("merge-base lookup should not run"),
    )

    candidate = precompiled.determine_wheel(
        tmp_path,
        location=str(wheel),
        commit="f" * 40,
    )

    assert candidate.url_or_path == str(wheel)


def test_invalid_commit_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(precompiled.PrecompiledWheelError, match="full 40-character"):
        precompiled.determine_wheel(tmp_path, commit="deadbeef")


def test_get_base_commit_uses_origin_main(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    (tmp_path / "file").write_text("main\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "file"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "main"], check=True)
    main_commit = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(
        ["git", "-C", str(tmp_path), "update-ref", "refs/remotes/origin/main", main_commit],
        check=True,
    )
    subprocess.run(["git", "-C", str(tmp_path), "commit", "--allow-empty", "-qm", "python change"], check=True)

    assert precompiled.get_base_commit_in_main(tmp_path) == main_commit


def test_extracts_allowlisted_files_and_preserves_tracked_interfaces(tmp_path: Path) -> None:
    package = tmp_path / "aphrodite"
    flash_attn = package / "vllm_flash_attn"
    flash_attn.mkdir(parents=True)
    interface = flash_attn / "flash_attn_interface.py"
    interface.write_text("tracked\n")

    first = _make_wheel(
        tmp_path / WHEEL_NAME,
        {
            "aphrodite/_C_stable_libtorch.abi3.so": b"first",
            "aphrodite/vllm_flash_attn/flash_attn_interface.py": b"wheel copy",
            "aphrodite/third_party/deep_gemm/old.py": b"old",
            "aphrodite/unrelated.py": b"ignore",
        },
    )
    patch = precompiled.extract_precompiled_wheel(tmp_path, first)

    assert (package / "_C_stable_libtorch.abi3.so").read_bytes() == b"first"
    assert interface.read_text() == "tracked\n"
    assert (package / "third_party/deep_gemm/old.py").is_file()
    assert not (package / "unrelated.py").exists()
    assert "aphrodite" in patch

    second_path = tmp_path / f"second-{WHEEL_NAME}"
    second = _make_wheel(
        second_path,
        {
            "aphrodite/_C_stable_libtorch.abi3.so": b"second",
            "aphrodite/third_party/deep_gemm/new.py": b"new",
        },
    )
    precompiled.extract_precompiled_wheel(tmp_path, second)

    assert (package / "_C_stable_libtorch.abi3.so").read_bytes() == b"second"
    assert not (package / "third_party/deep_gemm/old.py").exists()
    assert (package / "third_party/deep_gemm/new.py").is_file()
    manifest = json.loads((tmp_path / precompiled.MANIFEST_PATH).read_text())
    assert "aphrodite/third_party/deep_gemm/new.py" in manifest["paths"]


def test_checksum_mismatch_fails_before_replacing_existing_files(tmp_path: Path) -> None:
    target = tmp_path / "aphrodite/_C_stable_libtorch.abi3.so"
    target.parent.mkdir()
    target.write_bytes(b"existing")
    wheel_path = tmp_path / WHEEL_NAME
    candidate = _make_wheel(
        wheel_path,
        {"aphrodite/_C_stable_libtorch.abi3.so": b"replacement"},
    )
    candidate = precompiled.WheelCandidate(candidate.url_or_path, candidate.filename, "0" * 64)

    with pytest.raises(precompiled.PrecompiledWheelError, match="SHA-256 mismatch"):
        precompiled.extract_precompiled_wheel(tmp_path, candidate)

    assert target.read_bytes() == b"existing"


def test_zip_traversal_is_rejected(tmp_path: Path) -> None:
    candidate = _make_wheel(
        tmp_path / WHEEL_NAME,
        {"aphrodite/../escape.so": b"bad"},
    )

    with pytest.raises(precompiled.PrecompiledWheelError, match="Unsafe path"):
        precompiled.extract_precompiled_wheel(tmp_path, candidate)

    assert not (tmp_path / "escape.so").exists()


def test_unsafe_manifest_is_ignored(tmp_path: Path) -> None:
    manifest = tmp_path / precompiled.MANIFEST_PATH
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "url": "https://example.test/wheel.whl",
                "sha256": "a" * 64,
                "paths": ["aphrodite/../outside.so"],
            }
        )
    )

    assert precompiled._load_manifest(tmp_path) is None


def test_unsupported_host_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(precompiled.platform, "machine", lambda: "aarch64")

    with pytest.raises(precompiled.PrecompiledWheelError, match="Linux x86_64 CUDA only"):
        precompiled.determine_wheel(tmp_path, commit="f" * 40)
