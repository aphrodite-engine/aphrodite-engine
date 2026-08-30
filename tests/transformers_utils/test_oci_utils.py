"""``oci://`` model references resolve to a local path.

The scheme is explicit on purpose: a bare ``registry/name:tag`` is the same
shape as a HuggingFace repo id, so sniffing would hijack existing deployments.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from aphrodite.transformers_utils import llmman
from aphrodite.transformers_utils.oci_utils import (
    is_oci_uri,
    resolve_oci_model,
    strip_oci_scheme,
)


class TestScheme:
    def test_recognizes_the_oci_scheme(self):
        assert is_oci_uri("oci://ghcr.io/org/model:tag")
        assert is_oci_uri("OCI://ghcr.io/org/model:tag")

    @pytest.mark.parametrize(
        "value",
        [
            "meta-llama/Llama-3-8B",
            "ghcr.io/org/model:tag",
            "/local/path/to/model",
            "s3://bucket/key",
            "gs://bucket/key",
            "az://container/key",
            "",
            None,
        ],
    )
    def test_leaves_every_other_shape_alone(self, value):
        assert not is_oci_uri(value)

    def test_accepts_pathlib_input(self):
        assert not is_oci_uri(Path("/local/model"))

    def test_strips_the_scheme_only_when_present(self):
        assert strip_oci_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
        assert strip_oci_scheme("OCI://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
        assert strip_oci_scheme("meta-llama/Llama-3-8B") == "meta-llama/Llama-3-8B"


class TestResolveContract:
    """`llmman resolve --no-pull` reports where the daemon's pull landed."""

    def test_parses_the_documented_contract(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps({"reference": "r", "path": path, "format": "safetensors"})
            assert llmman.parse_resolve_output(line, "r") == path

    def test_tolerates_trailing_newline_and_leaked_diagnostics(self):
        with tempfile.TemporaryDirectory() as path:
            out = "pulling blobs...\n" + json.dumps({"path": path}) + "\n"
            assert llmman.parse_resolve_output(out, "r") == path

    def test_ignores_unknown_fields_so_the_contract_can_grow(self):
        with tempfile.TemporaryDirectory() as path:
            line = json.dumps({"path": path, "format": "gguf", "mmproj": "/x", "future": 1})
            assert llmman.parse_resolve_output(line, "r") == path

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   \n\n",
            "not json",
            '["a", "list"]',
            '{"no_path": 1}',
            '{"path": ""}',
            '{"path": 3}',
            '{"path": "/nonexistent/xyzzy"}',
        ],
    )
    def test_rejects_malformed_output(self, bad):
        with pytest.raises(RuntimeError):
            llmman.parse_resolve_output(bad, "r")


class TestEndpoint:
    @pytest.mark.parametrize(
        "host,want",
        [
            ("", "http://127.0.0.1:17434"),
            ("1.2.3.4:9999", "http://1.2.3.4:9999"),
            ("1.2.3.4", "http://1.2.3.4:17434"),
            ("http://1.2.3.4:9999/ignored", "http://1.2.3.4:9999"),
            # A wildcard bind is meaningful to the server but not to a client.
            ("0.0.0.0:9999", "http://127.0.0.1:9999"),
            ("[::]:9999", "http://[::1]:9999"),
        ],
    )
    def test_parses_every_llmman_host_form(self, host, want):
        with mock.patch.dict(os.environ, {llmman.HOST_ENV: host}):
            assert llmman.endpoint() == want

    def test_binary_default_and_override(self):
        with mock.patch.dict(os.environ, {llmman.BIN_ENV: ""}):
            assert llmman.llmman_bin() == "llmman"
        with mock.patch.dict(os.environ, {llmman.BIN_ENV: "/opt/llmman"}):
            assert llmman.llmman_bin() == "/opt/llmman"


class TestResolveOciModel:
    def test_rejects_an_empty_reference_without_touching_the_daemon(self):
        for ref in ("oci://", "oci://   "):
            with pytest.raises(ValueError):
                resolve_oci_model(ref)

    def test_strips_the_scheme_before_handing_off_to_llmman(self):
        with mock.patch(
            "aphrodite.transformers_utils.oci_utils.llmman.pull_and_resolve",
            return_value="/resolved",
        ) as acquire:
            assert resolve_oci_model("oci://ghcr.io/org/model:tag") == "/resolved"
        assert acquire.call_args[0][0] == "ghcr.io/org/model:tag"
        assert acquire.call_args[1]["progress"] is not None

    def test_reports_a_missing_binary(self):
        with (
            mock.patch.dict(os.environ, {llmman.BIN_ENV: "/definitely/not/here"}),
            pytest.raises(RuntimeError, match="not found"),
        ):
            llmman.resolve("ref")
