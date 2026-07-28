"""Parsing, validation and evaluation for task-parameter expressions.

Expressions may contain text, references such as ``{edge.field}``, function
calls and pipes. Expression evaluation uses ordinary Python dictionaries and
lists. Persisted models freeze the result after evaluation.
"""

import functools
import inspect
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from dagabaaz.constants import (
    MAX_EXPRESSION_EVALUATION_STEPS,
    MAX_EXPRESSION_LENGTH,
    MAX_EXPRESSION_RESULT_BYTES,
    MAX_PIPE_ARGUMENT_LENGTH,
)
from dagabaaz.json import (
    JsonValue,
    JsonWireValue,
    bounded_json_size,
    freeze_json,
    thaw_json,
)
from dagabaaz.models import ExpressionError
from dagabaaz.pipes import (
    BUILTIN_PIPES,
    PIPE_ARITY,
    _bounded_text_join,
)


@dataclass(frozen=True, slots=True)
class PipeCall:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipeSpec:
    """``argument_names`` lists arguments supplied after the piped value."""

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
    value: str  # Literal text, a reference or a function name.
    pipes: tuple[PipeCall, ...]
    refs: tuple[str, ...] = ()  # Function argument references.


Lookup = Callable[[str, str], object | None]


@dataclass(slots=True)
class EvaluationBudget:
    """Cumulative JSON bytes processed while resolving one node launch."""

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("evaluation byte limit must be a non-negative integer")
        if type(self.used) is not int or self.used < 0 or self.used > self.limit:
            raise ValueError("used evaluation bytes must be within the limit")

    def charge_size(self, size: int) -> None:
        if type(size) is not int or size < 0:
            raise ValueError("evaluation size must be a non-negative integer")
        if size > self.limit - self.used:
            raise ExpressionError(f"binding evaluation exceeds the {self.limit}-byte launch limit")
        self.used += size

    def charge(self, value: JsonValue | JsonWireValue) -> None:
        self.charge_size(_ensure_result_size(value))


# Function calls are recognised before references because a call has no
# top-level namespace separator.
_FUNC_CALL_RE = re.compile(r"^([a-zA-Z_]\w*)\(")


def _list_function(
    token: Token,
    lookup: Lookup,
    budget: EvaluationBudget | None,
) -> object:
    """References are added only while the encoded array remains within bounds."""

    result: list[JsonWireValue] = []
    size = 2
    for ref in token.refs:
        value = _resolve_reference(ref, lookup, budget)
        item_size = _result_size(value)
        candidate = size + item_size + (1 if result else 0)
        _check_result_size(candidate)
        result.append(value)
        size = candidate
    return result


_FUNCTION_DISPATCH: dict[
    str,
    Callable[[Token, Lookup, EvaluationBudget | None], object],
] = {
    "list": _list_function,
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
    return ExpressionVocabulary(pipes=tuple(pipe_specs), functions=tuple(sorted(_KNOWN_FUNCTIONS)))


def _token_evaluation_steps(tokens: tuple[Token, ...]) -> int:
    return sum(
        len(token.pipes)
        + (1 if token.kind == "ref" else len(token.refs) if token.kind == "call" else 0)
        for token in tokens
    )


@functools.lru_cache(maxsize=256)
def tokenize_expression(expression: str) -> tuple[Token, ...]:
    """The expression is parsed into immutable tokens.

    Malformed braces, references, calls and pipes raise ``ExpressionError``.
    Results are cached because one expression may be evaluated for several
    plans and ``Token`` is immutable.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f"expression is {len(expression)} characters; maximum is {MAX_EXPRESSION_LENGTH}"
        )

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
                    # A second opening brace would otherwise become text inside
                    # a flat reference.
                    raise ExpressionError(f"nested braces are not supported at position {scan}")
                if expression[scan] == "}":
                    scan += 1
                    break
                scan += 1
            else:
                # The closing brace was not found before the end of the expression.
                msg = f"Unclosed '{{' at position {pos}"
                raise ExpressionError(msg)

            ref_body = expression[pos + 1 : scan - 1].strip()
            if not ref_body:
                msg = f"Empty reference '{{}}' at position {pos}"
                raise ExpressionError(msg)

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

    result = tuple(tokens)
    evaluation_steps = _token_evaluation_steps(result)
    if evaluation_steps > MAX_EXPRESSION_EVALUATION_STEPS:
        raise ExpressionError(
            f"expression requires {evaluation_steps} reference and pipe evaluations; "
            f"maximum is {MAX_EXPRESSION_EVALUATION_STEPS}"
        )
    return result


def expression_evaluation_steps(expression: str) -> int:
    """Return the bounded number of reference lookups and pipe calls."""

    return _token_evaluation_steps(tokenize_expression(expression))


def _split_on_pipes(body: str, pos: int) -> list[str]:
    """Pipe separators at parenthesis depth zero divide the body.

    Reference and function-call parsers use the same rule.
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
    """Each raw pipe string becomes a ``PipeCall``."""
    pipes: list[PipeCall] = []
    for raw_pipe in raw_parts:
        if not raw_pipe:
            raise ExpressionError(f"Empty pipe in chain at position {pos}")
        pipes.append(_parse_pipe_call(raw_pipe, pos))
    return pipes


def _parse_ref_body(body: str, pos: int) -> tuple[str, list[PipeCall]]:
    """Reference parsing preserves the ordered pipe calls."""
    parts = _split_on_pipes(body, pos)

    ref = parts[0]
    if not ref:
        msg = f"Empty reference at position {pos}"
        raise ExpressionError(msg)
    _validate_ref(ref, pos)

    return ref, _parse_pipe_chain(parts[1:], pos)


def _parse_func_call(body: str, pos: int) -> Token:
    """Function-call parsing yields a token with optional pipes.

    Parenthesis depth identifies the end of the outer call when a later pipe
    also has arguments. Arguments are ``namespace.key`` references; nested
    function calls are not supported.
    """
    paren_start = body.index("(")
    func_name = body[:paren_start].strip()

    # The matching parenthesis belongs to the outer function call; later pipe
    # calls may contain their own parentheses.
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
        raise ExpressionError(f"Unclosed '(' in function '{func_name}' at position {pos}")

    # Arguments inside the outer parentheses are comma-separated references.
    args_str = body[paren_start + 1 : scan]
    refs: list[str] = [arg.strip() for arg in args_str.split(",") if arg.strip()]
    if len(refs) > MAX_EXPRESSION_EVALUATION_STEPS:
        raise ExpressionError(
            f"function {func_name!r} has {len(refs)} references; "
            f"maximum is {MAX_EXPRESSION_EVALUATION_STEPS}"
        )

    for ref in refs:
        if "(" in ref or ")" in ref:
            raise ExpressionError(f"Nested function calls are not supported at position {pos}")
        _validate_ref(ref, pos)

    # Text after the outer parentheses forms the pipe chain.
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
    """Commas divide pipe arguments, and a backslash escapes a comma.

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


def _validate_ref(ref: str, pos: int) -> None:
    """A reference contains a non-empty namespace and key."""

    dot = ref.find(".")
    if dot == -1:
        raise ExpressionError(
            f"Invalid reference '{ref}' at position {pos}: expected 'namespace.key'"
        )
    if dot == 0:
        raise ExpressionError(
            f"Invalid reference '{ref}' at position {pos}: namespace cannot be empty"
        )
    if dot == len(ref) - 1:
        raise ExpressionError(f"Invalid reference '{ref}' at position {pos}: key cannot be empty")


def _parse_pipe_call(raw: str, pos: int) -> PipeCall:
    """Pipe syntax produces a call for ``name`` or ``name(arg1,arg2)``."""
    paren_idx = raw.find("(")
    if paren_idx == -1:
        return PipeCall(name=raw.strip(), args=())

    name = raw[:paren_idx].strip()
    if not raw.endswith(")"):
        msg = f"Unclosed '(' in pipe '{name}' at position {pos}"
        raise ExpressionError(msg)

    args_str = raw[paren_idx + 1 : -1]
    args = _split_pipe_args(args_str)
    for argument in args:
        if len(argument) > MAX_PIPE_ARGUMENT_LENGTH:
            raise ExpressionError(
                f"pipe {name!r} argument is {len(argument)} characters; "
                f"maximum is {MAX_PIPE_ARGUMENT_LENGTH}"
            )
    return PipeCall(name=name, args=args)


def resolve_expression(
    expression: str,
    lookup: Lookup,
    *,
    budget: EvaluationBudget | None = None,
) -> JsonWireValue:
    """The lookup supplies values for expression evaluation.

    A single reference or call returns its routing value, including the result
    of any pipes. When the expression also contains text, resolved values are
    interpolated into a string. A missing standalone reference returns
    ``None``; before a pipe or within text it becomes an empty string.
    """
    tokens = tokenize_expression(expression)
    if not tokens:
        return None

    # A single reference or call preserves its routing value.
    if len(tokens) == 1 and tokens[0].kind in ("ref", "call"):
        token = tokens[0]
        value: JsonWireValue
        if token.kind == "ref":
            value = _resolve_reference(token.value, lookup, budget)
        else:
            value = _resolve_call(token, lookup, budget)
        if token.pipes:
            return _apply_pipes(value, token.pipes, budget)
        return value

    result = _bounded_text_join(_interpolation_parts(tokens, lookup, budget))
    if budget is not None:
        budget.charge(result)
    return result if result else None


def _interpolation_parts(
    tokens: tuple[Token, ...],
    lookup: Lookup,
    budget: EvaluationBudget | None,
) -> Iterator[str]:
    """Resolve interpolation in order so the size guard can stop evaluation."""

    for token in tokens:
        if token.kind == "text":
            yield token.value
        elif token.kind == "ref":
            value = _resolve_reference(token.value, lookup, budget)
            if token.pipes:
                value = _apply_pipes(value, token.pipes, budget)
            yield _interpolate(value)
        elif token.kind == "call":
            value = _resolve_call(token, lookup, budget)
            if token.pipes:
                value = _apply_pipes(value, token.pipes, budget)
            yield _interpolate(value)


def _resolve_call(
    token: Token,
    lookup: Lookup,
    budget: EvaluationBudget | None,
) -> JsonWireValue:
    """Registered functions return JSON values in ordinary Python containers."""
    handler = _FUNCTION_DISPATCH.get(token.value)
    if handler is None:
        raise ExpressionError(f"Unknown function '{token.value}'")
    value = handler(token, lookup, budget)
    try:
        result = thaw_json(freeze_json(value))
    except ValueError as exc:
        raise ExpressionError(f"function {token.value!r} returned a non-JSON value: {exc}") from exc
    size = _ensure_result_size(result)
    if budget is not None:
        budget.charge_size(size)
    return result


def _resolve_reference(
    ref: str,
    lookup: Lookup,
    budget: EvaluationBudget | None,
) -> JsonWireValue:
    """Lookup values are copied and validated as ordinary JSON values."""

    namespace, key = _split_ref(ref)
    try:
        result = thaw_json(freeze_json(lookup(namespace, key)))
    except ValueError as exc:
        raise ExpressionError(f"reference {ref!r} is not JSON-compatible: {exc}") from exc
    size = _ensure_result_size(result)
    if budget is not None:
        budget.charge_size(size)
    return result


def _interpolate(value: JsonWireValue) -> str:
    """Values embedded in text use their string representation.

    Arrays are comma-separated and omit null members.
    """

    if value is None:
        return ""
    if isinstance(value, list):
        return _bounded_text_join((str(item) for item in value if item is not None), ", ")
    return str(value)


def _split_ref(ref: str) -> tuple[str, str]:
    """The first dot separates ``namespace`` from ``key``."""
    dot = ref.find(".")
    if dot == -1:
        raise ExpressionError(f"Invalid reference '{ref}': expected 'namespace.key'")
    if dot == 0:
        raise ExpressionError(f"Invalid reference '{ref}': namespace cannot be empty")
    key = ref[dot + 1 :]
    if not key:
        raise ExpressionError(f"Invalid reference '{ref}': key cannot be empty")
    return ref[:dot], key


def _apply_pipes(
    value: JsonWireValue,
    pipes: tuple[PipeCall, ...],
    budget: EvaluationBudget | None,
) -> JsonWireValue:
    """Pipe calls run in order.

    A missing value becomes an empty string before the next call, which also
    allows the ``default`` pipe to select its fallback.
    """
    for pipe_call in pipes:
        fn = BUILTIN_PIPES.get(pipe_call.name)
        if fn is None:
            msg = f"Unknown pipe '{pipe_call.name}'"
            raise ExpressionError(msg)
        if value is None:
            value = ""
        try:
            pipe_result = fn(value, *pipe_call.args)
        except TypeError as exc:
            msg = f"Pipe '{pipe_call.name}' called with {len(pipe_call.args)} arg(s): {exc}"
            raise ExpressionError(msg) from exc
        try:
            value = thaw_json(freeze_json(pipe_result))
        except ValueError as exc:
            raise ExpressionError(
                f"pipe {pipe_call.name!r} returned a non-JSON value: {exc}"
            ) from exc
        size = _ensure_result_size(value)
        if budget is not None:
            budget.charge_size(size)
    return value


def _result_size(value: JsonValue | JsonWireValue) -> int:
    """Return the exact JSON size, stopping once the result exceeds its budget."""

    try:
        return bounded_json_size(value, MAX_EXPRESSION_RESULT_BYTES)
    except ValueError as exc:
        raise ExpressionError(f"expression result is not valid JSON: {exc}") from exc


def _check_result_size(size: int) -> None:
    if size > MAX_EXPRESSION_RESULT_BYTES:
        raise ExpressionError(
            f"expression result exceeds maximum size {MAX_EXPRESSION_RESULT_BYTES} bytes"
        )


def _ensure_result_size(value: JsonValue | JsonWireValue) -> int:
    size = _result_size(value)
    _check_result_size(size)
    return size


def validate_expression(expression: str) -> str | None:
    """Valid syntax yields ``None``; invalid syntax yields an error message."""
    try:
        tokens = tokenize_expression(expression)
    except ExpressionError as exc:
        return str(exc)

    for token in tokens:
        if token.kind == "text":
            continue
        for pipe_call in token.pipes:
            if pipe_call.name not in BUILTIN_PIPES:
                return f"Unknown pipe '{pipe_call.name}'"
            # Pipe arity can be checked without evaluating the expression.
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
    """Reference extraction returns edge names and run-input keys.

    Invalid syntax raises ``ExpressionError``.
    """
    tokens = tokenize_expression(expression)

    edge_names: set[str] = set()
    runtime_keys: set[str] = set()

    def _collect_ref(ref: str) -> None:
        namespace, key = _split_ref(ref)
        if namespace == "input":
            runtime_keys.add(key)
        else:
            edge_names.add(namespace)

    for token in tokens:
        if token.kind == "ref":
            _collect_ref(token.value)
        elif token.kind == "call":
            for ref in token.refs:
                _collect_ref(ref)

    return edge_names, runtime_keys
