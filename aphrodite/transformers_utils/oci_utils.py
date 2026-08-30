"""Resolve ``oci://`` model references to a local path.

A model published as a CNCF ModelPack (https://github.com/modelpack/model-spec)
artifact lives in an ordinary container registry, so it reuses the registry,
credentials, mirroring and air-gap tooling a deployment already has for
container images.

Registry work is delegated to the ``llmman`` CLI
(https://github.com/llmmanorg/llmman) rather than reimplemented here: it already
speaks the ModelPack media types, registry auth and resumable blob download, and
keeps a content-addressed local store. ``llmman resolve <reference>`` pulls the
image if it is not already local, extracts it, and prints one line of JSON on
stdout::

    {"reference": "ghcr.io/org/model:tag", "path": "/abs/path", "format": "safetensors"}

Only ``path`` is consumed; that directory is handed to the ordinary HuggingFace
loading path, exactly as if a local directory had been passed.

An explicit ``oci://`` scheme is required rather than sniffing a bare
``registry/name:tag``: that shape is indistinguishable from a HuggingFace repo
id (``org/model``), so guessing would silently hijack existing deployments.
"""

from pathlib import Path

from aphrodite.logger import init_logger
from aphrodite.transformers_utils import llmman

logger = init_logger(__name__)

SUPPORTED_SCHEMES = ["oci://"]


def is_oci_uri(model_or_path: str | Path | None) -> bool:
    """Whether the reference carries the ``oci://`` scheme.

    Cast to str to handle pathlib.Path inputs, mirroring is_runai_obj_uri.
    """
    if not model_or_path:
        return False
    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))


def strip_oci_scheme(reference: str | Path) -> str:
    """Drop the ``oci://`` prefix, leaving the bare registry reference."""
    text = str(reference)
    if is_oci_uri(text):
        return text[len(SUPPORTED_SCHEMES[0]) :]
    return text


def resolve_oci_model(reference: str | Path) -> str:
    """Pull an ``oci://`` reference through llmman and return the local path."""
    bare = strip_oci_scheme(reference)
    if not bare.strip():
        raise ValueError(f"empty OCI model reference: {reference!r}")

    def _progress(status, completed, total):
        if total:
            logger.info("llmman: %s (%s/%s bytes)", status, completed, total)
        else:
            logger.info("llmman: %s", status)

    return llmman.pull_and_resolve(bare.strip(), progress=_progress)
