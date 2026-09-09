# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from collections.abc import Sequence
from typing import Any, TypeVar, cast, overload

from aphrodite.utils.argparse_utils import FlexibleArgumentParser

UNSET = object()
_NamespaceT = TypeVar("_NamespaceT")
_GroupT = TypeVar("_GroupT", bound=argparse._ArgumentGroup)


def build_shadow_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build kwargs for the shadow argument with an ``UNSET`` default.

    Actions that mutate their default in place (append/extend/append_const/
    count) would crash on the bare ``UNSET`` sentinel, so they are remapped to
    an equivalent store-style action; the shadow value only needs to flip away
    from ``UNSET`` when the arg is passed explicitly.
    """
    shadow_kwargs = {**kwargs, "default": UNSET}
    action = kwargs.get("action")

    if action in ("append", "extend"):
        shadow_kwargs["action"] = "store"

    elif action in ("append_const", "count"):
        shadow_kwargs["action"] = "store_const"

        if action == "count":
            shadow_kwargs["const"] = True

    return shadow_kwargs


class TrackingNamespace(argparse.Namespace):
    """Proxy that wraps an argparse namespace with explicit keys, which
    can be filtered down to a dict containing only explicitly passed values.
    """

    def __init__(self, unfiltered_ns: argparse.Namespace, explicit_keys: frozenset[str]) -> None:
        # We never have nested tracking namespaces, but explicitly guard
        # against them to prevent bad behavior with nested __dict__ overrides.
        if isinstance(unfiltered_ns, TrackingNamespace):
            raise ValueError("Tracking namespaces cannot be nested")

        self.unfiltered_ns = unfiltered_ns
        self.explicit_keys = explicit_keys

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("unfiltered_ns", "explicit_keys"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.unfiltered_ns, name, value)

    def get_explicit_kwargs_dict(self):
        """Return a dict containing only the explicitly passed key-value pairs."""
        return {k: v for k, v in vars(self.unfiltered_ns).items() if k in self.explicit_keys}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.unfiltered_ns, name)

    @property
    def __dict__(self):
        # NOTE: We do this so that vars() etc forward directly into the encapsulated namespace,
        # which makes this class a drop-in replacement for the original namespace, while also
        # ensuring that updates to the encapsulated namespace are correctly reflected.
        return self.unfiltered_ns.__dict__


def _track_group(real_group: _GroupT, shadow_group: argparse._ArgumentGroup) -> _GroupT:
    add_real_argument = real_group.add_argument

    def add_argument(*args: Any, **kwargs: Any) -> argparse.Action:
        action = add_real_argument(*args, **kwargs)
        default_kwargs = build_shadow_kwargs(kwargs)
        shadow_group.add_argument(*args, **default_kwargs)
        return action

    setattr(real_group, "add_argument", add_argument)
    return real_group


def _track_subparsers(
    real_sub: argparse._SubParsersAction, shadow_sub: argparse._SubParsersAction
) -> argparse._SubParsersAction:
    add_real_parser = real_sub.add_parser

    def add_parser(name, *args, **kwargs):
        real_parser = add_real_parser(name, *args, **kwargs)
        # real_parser is a TrackingArgumentParser with its own _shadow.
        # Reuse that shadow as the parent shadow's child — so when
        # real_parser.add_argument() mirrors to real_parser._shadow,
        # the parent's shadow sees it too.
        shadow_sub._name_parser_map[name] = real_parser._shadow
        return real_parser

    setattr(real_sub, "add_parser", add_parser)
    return real_sub


class TrackingArgumentParser(FlexibleArgumentParser):
    """Drop-in replacement for FlexibleArgumentParser, which tracks keys that
    were explicitly passed as args on the parser namespace.

    Unfortunately, Argparse does not provide an easy way of doing this without
    depending on a lot of internal attributes, so we implement it by instead
    using a 'shadow' parser, which is essentially a clone of the parser, where
    defaults are overridden to `None`. By comparing the parser against its
    shadow, we can tell which values were passed in a non-destructive manner.
    """

    def __init__(self, *args, **kwargs):
        # NOTE: We have to define the shadow parser before calling init,
        # with add_help=False, since otherwise init will call add_argument
        # and delegate to the override on this class and cause problems.
        shadow_kwargs = {**kwargs, "add_help": False}
        self._shadow = FlexibleArgumentParser(*args, **shadow_kwargs)
        super().__init__(*args, **kwargs)

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action:
        """Add an arg to the parser & the shadow, where the latter has UNSET for the default."""
        action = super().add_argument(*args, **kwargs)
        shadow_kwargs = build_shadow_kwargs(kwargs)
        self._shadow.add_argument(*args, **shadow_kwargs)
        return action

    def add_argument_group(self, *args, **kwargs) -> argparse._ArgumentGroup:
        real_group = super().add_argument_group(*args, **kwargs)
        shadow_group = self._shadow.add_argument_group(*args, **kwargs)
        return _track_group(real_group, shadow_group)

    def add_mutually_exclusive_group(self, *args, **kwargs) -> argparse._MutuallyExclusiveGroup:
        real_group = super().add_mutually_exclusive_group(*args, **kwargs)
        shadow_group: argparse._MutuallyExclusiveGroup = self._shadow.add_mutually_exclusive_group(*args, **kwargs)
        return _track_group(real_group, shadow_group)

    def add_subparsers(self, *args, **kwargs) -> argparse._SubParsersAction:
        real_sub = super().add_subparsers(*args, **kwargs)
        shadow_sub = self._shadow.add_subparsers(*args, **kwargs)
        return _track_subparsers(real_sub, shadow_sub)

    def build_tracking_namespace(self, real_ns: argparse.Namespace, shadow_ns: argparse.Namespace) -> TrackingNamespace:
        """Build a tracking namespace for the real / shadow namespaces."""
        explicit_keys = frozenset(k for k, v in vars(shadow_ns).items() if v is not UNSET)
        return TrackingNamespace(real_ns, explicit_keys)

    @overload
    def parse_args(self, args: Sequence[str] | None = None, namespace: None = None) -> TrackingNamespace: ...

    @overload
    def parse_args(self, args: Sequence[str] | None, namespace: _NamespaceT) -> _NamespaceT: ...

    @overload
    def parse_args(self, *, namespace: _NamespaceT) -> _NamespaceT: ...

    @overload
    def parse_args(
        self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace: ...

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: _NamespaceT | None = None,
    ) -> TrackingNamespace | _NamespaceT:
        """Parse the args on the real/shadow parser."""
        # Only the real parser should use the namespace if one is,
        # given since shadow parser will set its own defaults to None.
        real_ns = super().parse_args(
            list(args) if args is not None else None,
            cast(argparse.Namespace | None, namespace),
        )
        if namespace is not None:
            return namespace
        shadow_ns = self._shadow.parse_args(args)
        if real_ns is None or shadow_ns is None:
            raise ValueError("Parse args created empty namespaces")

        # If this is called through parse_known_args on self, we will already
        # get a TrackingNamespace back, which will already have set the explicit
        # keys through build_tracking_namespace, so no need to do it again.
        if isinstance(real_ns, TrackingNamespace):
            return real_ns

        return self.build_tracking_namespace(real_ns, shadow_ns)

    @overload
    def parse_known_args(
        self, args: Sequence[str] | None = None, namespace: None = None
    ) -> tuple[TrackingNamespace, list[str]]: ...

    @overload
    def parse_known_args(self, args: Sequence[str] | None, namespace: _NamespaceT) -> tuple[_NamespaceT, list[str]]: ...

    @overload
    def parse_known_args(self, *, namespace: _NamespaceT) -> tuple[_NamespaceT, list[str]]: ...

    def parse_known_args(
        self,
        args: Sequence[str] | None = None,
        namespace: _NamespaceT | None = None,
    ) -> tuple[TrackingNamespace | _NamespaceT, list[str]]:
        """Parse the known args on the real/shadow parser."""
        if namespace is not None:
            _, remaining = super().parse_known_args(args, namespace)
            return namespace, remaining
        real_ns, remaining = super().parse_known_args(args, None)
        shadow_ns, _ = self._shadow.parse_known_args(args)
        tracked_ns = self.build_tracking_namespace(real_ns, shadow_ns)

        return tracked_ns, remaining
