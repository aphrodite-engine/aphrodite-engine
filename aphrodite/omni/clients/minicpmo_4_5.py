# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Client-side duplex session preset for MiniCPM-o 4.5 native duplex."""

from __future__ import annotations

from dataclasses import replace

from typing_extensions import Unpack

from aphrodite.omni.clients.duplex import SessionConfig, SessionConfigOverrides

__all__ = ["create_duplex_session_config"]


def create_duplex_session_config(**overrides: Unpack[SessionConfigOverrides]) -> SessionConfig:
    """Session preset matching the MiniCPM-o 4.5 native duplex deployment."""
    extra_body: dict[str, object] = {"native_duplex": True, "force_listen_count": 0}
    extra_body.update(overrides.pop("extra_body", {}))
    config = SessionConfig(
        overlap_policy="listen_only",
        playback_commit_policy="ack_only",
        extra_body=extra_body,
    )
    return replace(config, **overrides) if overrides else config
