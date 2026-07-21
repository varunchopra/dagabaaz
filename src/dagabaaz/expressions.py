"""Expression evaluation engine for ``{namespace.key | pipe}`` bindings.

Three layers, each callable independently:

  tokenize_expression() -- string -> token list (parsing)
  resolve_expression()  -- tokens + lookup -> value (evaluation)
  validate_expression() -- tokens -> error string (static analysis)

Supports function calls: ``{list(ns1.key, ns2.key)}`` produces a list
from multiple variable references. Functions are pre-pipe (the result
flows into the pipe chain): ``{list(a.url, b.url) | join(,)}``.
"""

import functools
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from dagabaaz.models import ExpressionError
from dagabaaz.pipes import BUILTIN_PIPES, PIPE_ARITY


@dataclass(frozen=True, slots=True)
class PipeCall:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipeSpec:
    """Argument names exclude the implicit piped value."""

    name: str
    min_args: int
    max_args: int
    argument_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpressionVocabulary:
    pipes: tuple[PipeSpec, ...]
    functions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Token:
    kind: Literal["text", "ref", "call"]
    value: str  # literal text, "namespace.key", or function name
    pipes: tuple[PipeCall, ...]
    refs: tuple[str, ...] = ()  # argument refs for "call" tokens


Lookup = Callable[[str, str], object | None]

# Detects function call syntax: valid identifier followed by opening paren.
# Must match before _parse_ref_body runs — function calls don't contain a
# dot at the top level, so they'd fail the "namespace.key" validation.
_FUNC_CALL_RE = re.compile(r"^([a-zA-Z_]\w*)\(")

_FUNCTION_DISPATCH: dict[str, Callable[[Token, Lookup], object]] = {
    "list": lambda token, lookup: [lookup(*_split_ref(ref)) for ref in token.refs],
}

_KNOWN_FUNCTIONS = frozenset(_FUNCTION_DISPATCH.keys())


def get_expression_vocabulary() -> ExpressionVocabulary:
    pipe_specs: list[PipeSpec] = []
    for name, pipe in BUILTIN_PIPES.items():
        min_args, max_args = PIPE_ARITY[name]
        pipe_specs.append(
            PipeSpec(
                name=name,
                min_args=min_args,
                max_args=max_args,
                argument_names=tuple(inspect.signature(pipe).parameters)[1:],
            )
        )
    return ExpressionVocabulary(
        pipes=tuple(pipe_specs), functions=tuple(sorted(_KNOWN_FUNCTIONS))
    )


@functools.lru_cache(maxsize=256)
def tokenize_expression(expression: str) -> tuple[Token, ...]:
    """Parse an expression string into tokens.

    Walks left to right, switching between TEXT and REF modes on ``{``/``}``.
    Raises ExpressionError on malformed input (unclosed braces, empty refs).

    Cached because resolve_expression re-tokenizes on every call — strings
    are hashable and the return is an immutable tuple of frozen dataclasses,
    so cached values cannot be corrupted by callers.
    """
    tokens: list[Token] = []
    pos = 0
    length = len(expression)
    buf: list[str] = []

    while pos < length:
        char = expression[pos]

        if char == "\\" and pos + 1 < length and expression[pos + 1] in "{}":
            buf.append(expression[pos + 1])
            pos += 2
            continue

        if char == "{":
            if buf:
                tokens.append(Token(kind="text", value="".join(buf), pipes=()))
                buf.clear()

            scan = pos + 1
            while scan < length:
                if expression[scan] == "{":
                    # Nested expressions are not supported — _parse_ref_body
                    # treats the body as flat text, so nested braces would
                    # silently misparse rather than fail.
                    raise ExpressionError(
                        f"Nested braces not supported at position {scan}"
                    )
                if expression[scan] == "}":
                    scan += 1
                    break
                scan += 1
            else:
                # Reached end of string without finding closing brace
                msg = f"Unclosed '{{' at position {pos}"
                raise ExpressionError(msg)

            ref_body = expression[pos + 1 : scan - 1].strip()
            if not ref_body:
                msg = f"Empty reference '{{}}' at position {pos}"
                raise ExpressionError(msg)

            # Function call detection: must happen BEFORE _parse_ref_body
            # because function calls like "list(a.url, b.url)" don't have a
            # top-level dot and would fail the namespace.key validation.
            func_match = _FUNC_CALL_RE.match(ref_body)
            if func_match:
                token = _parse_func_call(ref_body, pos)
                tokens.append(token)
            else:
                ref, pipes = _parse_ref_body(ref_body, pos)
                tokens.append(Token(kind="ref", value=ref, pipes=tuple(pipes)))
            pos = scan

        elif char == "}":
            msg = f"Unexpected '}}' at position {pos}"
            raise ExpressionError(msg)

        else:
            buf.append(char)
            pos += 1

    if buf:
        tokens.append(Token(kind="text", value="".join(buf), pipes=()))

    return tuple(tokens)


def _split_on_pipes(body: str, pos: int) -> list[str]:
    """Split a body string on ``|`` at paren depth 0.

    Shared by _parse_ref_body (ref | pipes) and _parse_func_call (func(...) | pipes).
    """
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in body:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ExpressionError(f"Unbalanced closing paren at position {pos}")
            current.append(char)
        elif char == "|" and depth == 0:
            parts.append("".join(current).strip())
            current.clear()
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return parts


def _parse_pipe_chain(raw_parts: list[str], pos: int) -> list[PipeCall]:
    """Parse a list of raw pipe strings into PipeCall objects."""
    pipes: list[PipeCall] = []
    for raw_pipe in raw_parts:
        if not raw_pipe:
            raise ExpressionError(f"Empty pipe in chain at position {pos}")
        pipes.append(_parse_pipe_call(raw_pipe, pos))
    return pipes


def _parse_ref_body(body: str, pos: int) -> tuple[str, list[PipeCall]]:
    """Parse ``namespace.key | pipe1 | pipe2(arg)`` into ref + pipe list."""
    parts = _split_on_pipes(body, pos)

    ref = parts[0]
    if not ref:
        msg = f"Empty reference at position {pos}"
        raise ExpressionError(msg)
    if "." not in ref:
        msg = f"Invalid reference '{ref}' at position {pos}: expected 'namespace.key'"
        raise ExpressionError(msg)

    return ref, _parse_pipe_chain(parts[1:], pos)


def _parse_func_call(body: str, pos: int) -> Token:
    """Parse ``list(ns1.key, ns2.key) | pipe1 | pipe2`` into a call Token.

    Depth-tracked scan finds the matching ``)`` for the opening ``(``,
    then splits the remainder on ``|`` for the pipe chain.  Arguments
    are ``namespace.key`` refs only (v1 — no nested expressions).
    """
    paren_start = body.index("(")
    func_name = body[:paren_start].strip()

    # Depth-tracked scan to find matching closing paren.
    # Critically, this finds list()'s ) — NOT join(,)'s ).
    depth = 1
    scan = paren_start + 1
    while scan < len(body):
        if body[scan] == "(":
            depth += 1
        elif body[scan] == ")":
            depth -= 1
            if depth == 0:
                break
        scan += 1
    else:
        raise ExpressionError(
            f"Unclosed '(' in function '{func_name}' at position {pos}"
        )

    # Everything inside the matched parens = comma-separated refs.
    args_str = body[paren_start + 1 : scan]
    refs: list[str] = [arg.strip() for arg in args_str.split(",") if arg.strip()]

    for ref in refs:
        if "." not in ref:
            msg = (
                f"Invalid reference '{ref}' at position {pos}: expected 'namespace.key'"
            )
            raise ExpressionError(msg)

    # Everything after the matched ) = pipe chain.
    remainder = body[scan + 1 :].strip()
    pipes: list[PipeCall] = []
    if remainder:
        if not remainder.startswith("|"):
            raise ExpressionError(
                f"Expected '|' after function call at position {pos}, got '{remainder[0]}'"
            )
        pipe_parts = _split_on_pipes(remainder, pos)
        pipes = _parse_pipe_chain(pipe_parts[1:], pos)

    if func_name not in _KNOWN_FUNCTIONS:
        raise ExpressionError(f"Unknown function '{func_name}'")

    return Token(kind="call", value=func_name, pipes=tuple(pipes), refs=tuple(refs))


def _split_pipe_args(args_str: str) -> tuple[str, ...]:
    """Split pipe argument string on commas, supporting backslash-escaped commas.

    A literal comma in a pipe argument can be written as ``\\,``.
    For example: ``replace(foo\\,bar,baz)`` passes ``"foo,bar"`` and ``"baz"``.
    """
    args: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(args_str):
        if args_str[i] == "\\" and i + 1 < len(args_str) and args_str[i + 1] == ",":
            current.append(",")
            i += 2
        elif args_str[i] == ",":
            args.append("".join(current).strip())
            current.clear()
            i += 1
        else:
            current.append(args_str[i])
            i += 1
    remaining = "".join(current).strip()
    if remaining:
        args.append(remaining)
    return tuple(args)


def _parse_pipe_call(raw: str, pos: int) -> PipeCall:
    """Parse ``name`` or ``name(arg1,arg2)`` into a PipeCall."""
    paren_idx = raw.find("(")
    if paren_idx == -1:
        return PipeCall(name=raw.strip(), args=())

    name = raw[:paren_idx].strip()
    if not raw.endswith(")"):
        msg = f"Unclosed '(' in pipe '{name}' at position {pos}"
        raise ExpressionError(msg)

    args_str = raw[paren_idx + 1 : -1]
    args = _split_pipe_args(args_str)
    return PipeCall(name=name, args=args)


def resolve_expression(expression: str, lookup: Lookup) -> object | None:
    """Evaluate an expression string to a value.

    ``lookup(namespace, key)`` provides variable resolution — the expression
    engine knows nothing about artifacts, slugs, or configs.

    Resolution rules:
      1. Single ref/call, no pipes → native type (preserves lists, numbers)
      2. Single ref/call with pipes → pipe chain result
      3. Multiple tokens / mixed text → string interpolation
      4. Missing ref → None (no pipes) or "" (with pipes)
    """
    tokens = tokenize_expression(expression)
    if not tokens:
        return None

    # Rule 1 & 2: single token preserves native type.
    if len(tokens) == 1 and tokens[0].kind in ("ref", "call"):
        token = tokens[0]
        value: object
        if token.kind == "ref":
            namespace, key = _split_ref(token.value)
            value = lookup(namespace, key)
        else:
            value = _resolve_call(token, lookup)
        if token.pipes:
            return _apply_pipes(value, token.pipes)
        return value

    # Rule 3: multiple tokens → string interpolation.
    parts: list[str] = []
    for token in tokens:
        if token.kind == "text":
            parts.append(token.value)
        elif token.kind == "ref":
            namespace, key = _split_ref(token.value)
            value = lookup(namespace, key)
            if token.pipes:
                value = _apply_pipes(value, token.pipes)
            parts.append(str(value) if value is not None else "")
        elif token.kind == "call":
            value = _resolve_call(token, lookup)
            if token.pipes:
                value = _apply_pipes(value, token.pipes)
            # In interpolation context, render lists as comma-joined with
            # Nones filtered. Standalone {list(...)} preserves Nones for
            # positional correspondence; embedded in text, Nones are
            # confusing so we drop them.
            if isinstance(value, list):
                parts.append(", ".join(str(v) for v in value if v is not None))
            else:
                parts.append(str(value) if value is not None else "")

    result = "".join(parts)
    return result if result else None


def _resolve_call(token: Token, lookup: Lookup) -> object:
    """Resolve a function call token to its value."""
    handler = _FUNCTION_DISPATCH.get(token.value)
    if handler is None:
        raise ExpressionError(f"Unknown function '{token.value}'")
    return handler(token, lookup)


def _split_ref(ref: str) -> tuple[str, str]:
    """Split ``namespace.key`` into ``(namespace, key)``."""
    dot = ref.find(".")
    if dot == -1:
        raise ExpressionError(f"Invalid reference '{ref}': expected 'namespace.key'")
    key = ref[dot + 1 :]
    if not key:
        raise ExpressionError(f"Invalid reference '{ref}': key cannot be empty")
    return ref[:dot], key


def _apply_pipes(value: object, pipes: tuple[PipeCall, ...]) -> object:
    """Apply a chain of pipe functions to a value.

    None is converted to empty string before each pipe so string operations
    don't produce ``"None"``. The ``default`` pipe relies on this: empty
    string is falsy, triggering the fallback.
    """
    for pipe_call in pipes:
        fn = BUILTIN_PIPES.get(pipe_call.name)
        if fn is None:
            msg = f"Unknown pipe '{pipe_call.name}'"
            raise ExpressionError(msg)
        if value is None:
            value = ""
        try:
            value = fn(value, *pipe_call.args)
        except TypeError as exc:
            msg = f"Pipe '{pipe_call.name}' called with {len(pipe_call.args)} arg(s): {exc}"
            raise ExpressionError(msg) from exc
    return value


RESERVED_NAMESPACES = frozenset({"input", "config"})


def validate_expression(expression: str) -> str | None:
    """Check expression syntax. Returns error message or None on success."""
    try:
        tokens = tokenize_expression(expression)
    except ExpressionError as exc:
        return str(exc)

    for token in tokens:
        if token.kind == "text":
            continue
        # Validate pipes on both ref and call tokens.
        for pipe_call in token.pipes:
            if pipe_call.name not in BUILTIN_PIPES:
                return f"Unknown pipe '{pipe_call.name}'"
            # Arity validation — reject wrong argument count at save time.
            arity = PIPE_ARITY.get(pipe_call.name)
            if arity is not None:
                min_args, max_args = arity
                n = len(pipe_call.args)
                if n < min_args or n > max_args:
                    if min_args == max_args:
                        return f"Pipe '{pipe_call.name}' expects {min_args} arg(s), got {n}"
                    return f"Pipe '{pipe_call.name}' expects {min_args}-{max_args} arg(s), got {n}"

    return None


def extract_refs(expression: str) -> tuple[set[str], set[str]]:
    """Extract node slugs and runtime input keys from an expression.

    Returns ``(node_slugs, runtime_keys)``. Handles both ``ref`` tokens
    (single variable) and ``call`` tokens (function arguments).
    Raises ``ExpressionError`` on syntax errors.
    """
    tokens = tokenize_expression(expression)

    slugs: set[str] = set()
    runtime_keys: set[str] = set()

    def _collect_ref(ref: str) -> None:
        namespace, key = _split_ref(ref)
        if namespace == "input":
            runtime_keys.add(key)
        elif namespace != "config":
            slugs.add(namespace)

    for token in tokens:
        if token.kind == "ref":
            _collect_ref(token.value)
        elif token.kind == "call":
            for ref in token.refs:
                _collect_ref(ref)

    return slugs, runtime_keys
