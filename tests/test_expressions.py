from collections.abc import Callable

import pytest

from dagabaaz.expressions import (
    ExpressionError,
    PipeCall,
    Token,
    extract_refs,
    resolve_expression,
    tokenize_expression,
    validate_expression,
)
from dagabaaz.pipes import BUILTIN_PIPES as _PIPES
from dagabaaz.pipes import PIPE_ARITY


def test_all_pipes_have_arity_entries() -> None:
    """Every pipe in BUILTIN_PIPES must have a corresponding PIPE_ARITY entry."""
    assert set(PIPE_ARITY.keys()) == set(_PIPES.keys())


class TestTokenize:
    def test_plain_text(self) -> None:
        assert tokenize_expression("hello world") == (
            Token(kind="text", value="hello world", pipes=()),
        )

    def test_single_ref(self) -> None:
        assert tokenize_expression("{slug.key}") == (
            Token(kind="ref", value="slug.key", pipes=()),
        )

    def test_ref_with_pipe(self) -> None:
        assert tokenize_expression("{slug.key | upper}") == (
            Token(kind="ref", value="slug.key", pipes=(PipeCall("upper", ()),)),
        )

    def test_pipe_with_args(self) -> None:
        assert tokenize_expression("{slug.key | replace(old,new)}") == (
            Token(
                kind="ref",
                value="slug.key",
                pipes=(PipeCall("replace", ("old", "new")),),
            ),
        )

    def test_pipe_chain(self) -> None:
        tokens = tokenize_expression("{slug.key | trim | upper}")
        assert len(tokens) == 1
        assert tokens[0].pipes == (
            PipeCall("trim", ()),
            PipeCall("upper", ()),
        )

    def test_multiple_refs(self) -> None:
        tokens = tokenize_expression("{a.x} and {b.y}")
        assert len(tokens) == 3
        assert tokens[0] == Token(kind="ref", value="a.x", pipes=())
        assert tokens[1] == Token(kind="text", value=" and ", pipes=())
        assert tokens[2] == Token(kind="ref", value="b.y", pipes=())

    def test_mixed_text_and_refs(self) -> None:
        tokens = tokenize_expression("Hello {input.name}!")
        assert tokens == (
            Token(kind="text", value="Hello ", pipes=()),
            Token(kind="ref", value="input.name", pipes=()),
            Token(kind="text", value="!", pipes=()),
        )

    def test_escaped_braces(self) -> None:
        tokens = tokenize_expression("\\{not a ref\\}")
        assert tokens == (Token(kind="text", value="{not a ref}", pipes=()),)

    def test_empty_string(self) -> None:
        assert tokenize_expression("") == ()

    def test_unclosed_brace(self) -> None:
        with pytest.raises(ExpressionError, match="Unclosed"):
            tokenize_expression("{slug.key")

    def test_unexpected_closing_brace(self) -> None:
        with pytest.raises(ExpressionError, match="Unexpected"):
            tokenize_expression("text}")

    def test_empty_ref(self) -> None:
        with pytest.raises(ExpressionError, match="Empty reference"):
            tokenize_expression("{}")

    def test_ref_without_dot(self) -> None:
        with pytest.raises(ExpressionError, match="namespace.key"):
            tokenize_expression("{nodot}")

    def test_pipe_with_unclosed_paren(self) -> None:
        with pytest.raises(ExpressionError, match="Unclosed"):
            tokenize_expression("{slug.key | replace(old}")

    def test_whitespace_in_ref(self) -> None:
        tokens = tokenize_expression("{ slug.key | upper }")
        assert tokens[0].value == "slug.key"
        assert tokens[0].pipes == (PipeCall("upper", ()),)

    def test_multiple_pipes_with_args(self) -> None:
        tokens = tokenize_expression("{s.k | replace(a,b) | truncate(50)}")
        assert tokens[0].pipes == (
            PipeCall("replace", ("a", "b")),
            PipeCall("truncate", ("50",)),
        )


class TestEmptyPipeRejection:
    """Empty pipes in chain raise ExpressionError."""

    def test_trailing_pipe_rejected(self) -> None:
        with pytest.raises(ExpressionError, match="Empty pipe"):
            tokenize_expression("{slug.key | }")

    def test_double_pipe_rejected(self) -> None:
        with pytest.raises(ExpressionError, match="Empty pipe"):
            tokenize_expression("{slug.key | upper | | lower}")

    def test_leading_pipe_only_rejected(self) -> None:
        """Pipe with only whitespace in body."""
        with pytest.raises(ExpressionError, match="Empty pipe"):
            tokenize_expression("{slug.key |  }")


class TestBackslashEscapedCommas:
    """Backslash-escaped commas in pipe arguments."""

    def test_escaped_comma_in_single_arg(self) -> None:
        tokens = tokenize_expression("{s.k | replace(foo\\,bar,baz)}")
        pipe = tokens[0].pipes[0]
        assert pipe.name == "replace"
        assert pipe.args == ("foo,bar", "baz")

    def test_no_escape_normal_comma(self) -> None:
        tokens = tokenize_expression("{s.k | replace(a,b)}")
        pipe = tokens[0].pipes[0]
        assert pipe.args == ("a", "b")

    def test_escaped_comma_single_pipe_arg(self) -> None:
        tokens = tokenize_expression("{s.k | replace(foo\\,bar)}")
        pipe = tokens[0].pipes[0]
        assert pipe.args == ("foo,bar",)


class TestPipes:
    @pytest.mark.parametrize(
        "pipe,input_val,expected",
        [
            ("upper", "hello", "HELLO"),
            ("lower", "HELLO", "hello"),
            ("trim", "  hi  ", "hi"),
            ("title", "hello world", "Hello World"),
        ],
    )
    def test_basic_string_pipes(self, pipe: str, input_val: str, expected: str) -> None:
        assert _PIPES[pipe](input_val) == expected

    def test_replace(self) -> None:
        assert _PIPES["replace"]("hello world", "world", "earth") == "hello earth"

    def test_replace_delete(self) -> None:
        assert _PIPES["replace"]("hello.tmp", ".tmp") == "hello"

    def test_strip(self) -> None:
        assert _PIPES["strip"]("__hi__", "_") == "hi"

    def test_strip_default(self) -> None:
        assert _PIPES["strip"]("  hi  ") == "hi"

    def test_lstrip(self) -> None:
        assert _PIPES["lstrip"]("__hi__", "_") == "hi__"

    def test_rstrip(self) -> None:
        assert _PIPES["rstrip"]("__hi__", "_") == "__hi"

    def test_default_with_value(self) -> None:
        assert _PIPES["default"]("hello", "fallback") == "hello"

    def test_default_with_empty(self) -> None:
        assert _PIPES["default"]("", "fallback") == "fallback"

    def test_default_with_none(self) -> None:
        assert _PIPES["default"](None, "fallback") == "fallback"

    @pytest.mark.parametrize(
        "value,expected",
        [(0, 0), (False, False), ([], [])],
    )
    def test_default_preserves_falsy_non_empty(
        self, value: object, expected: object
    ) -> None:
        assert _PIPES["default"](value, "fallback") == expected

    def test_required_with_value(self) -> None:
        assert _PIPES["required"]("hello") == "hello"

    def test_required_with_empty(self) -> None:
        with pytest.raises(ExpressionError, match="Required"):
            _PIPES["required"]("")

    def test_required_passes_zero(self) -> None:
        assert _PIPES["required"](0) == 0

    def test_required_passes_false(self) -> None:
        assert _PIPES["required"](False) is False

    def test_first_list(self) -> None:
        assert _PIPES["first"](["a", "b", "c"]) == "a"

    def test_first_empty(self) -> None:
        assert _PIPES["first"]([]) is None

    def test_first_scalar(self) -> None:
        assert _PIPES["first"]("hello") == "hello"

    def test_last_list(self) -> None:
        assert _PIPES["last"](["a", "b", "c"]) == "c"

    def test_nth(self) -> None:
        assert _PIPES["nth"](["a", "b", "c"], "1") == "b"

    def test_nth_out_of_range(self) -> None:
        assert _PIPES["nth"](["a"], "5") is None

    def test_nth_non_integer_index(self) -> None:
        """Non-integer index returns None instead of crashing."""
        assert _PIPES["nth"](["a", "b"], "abc") is None

    def test_join(self) -> None:
        assert _PIPES["join"](["a", "b", "c"], ", ") == "a, b, c"

    def test_join_custom_sep(self) -> None:
        assert _PIPES["join"](["a", "b"], " | ") == "a | b"

    def test_join_scalar(self) -> None:
        assert _PIPES["join"]("hello", ", ") == "hello"

    def test_join_filters_none(self) -> None:
        """None items are excluded from join output."""
        assert _PIPES["join"]([None, "a", None, "b"], "-") == "a-b"

    def test_basename(self) -> None:
        assert _PIPES["basename"]("/data/output/file.txt") == "file.txt"

    def test_dirname(self) -> None:
        assert _PIPES["dirname"]("/data/output/file.txt") == "/data/output"

    def test_stem(self) -> None:
        assert _PIPES["stem"]("/data/file.txt") == "file"

    def test_ext(self) -> None:
        assert _PIPES["ext"]("/data/file.txt") == ".txt"

    def test_ext_no_extension(self) -> None:
        assert _PIPES["ext"]("Makefile") == ""

    def test_basename_backslash_path(self) -> None:
        """Backslash paths are treated as literal names (POSIX semantics)."""
        assert _PIPES["basename"]("a\\b\\c.txt") == "a\\b\\c.txt"

    def test_dirname_posix(self) -> None:
        """dirname uses POSIX path semantics regardless of host OS."""
        assert _PIPES["dirname"]("/data/output/file.txt") == "/data/output"

    def test_dirname_bare_filename(self) -> None:
        """Bare filename returns '.' under POSIX semantics (was '' with os.path)."""
        assert _PIPES["dirname"]("file.txt") == "."

    def test_urlencode(self) -> None:
        assert _PIPES["urlencode"]("hello world") == "hello%20world"

    def test_urldecode(self) -> None:
        assert _PIPES["urldecode"]("hello%20world") == "hello world"

    def test_int_from_string(self) -> None:
        assert _PIPES["int"]("42") == 42

    def test_int_from_float_string(self) -> None:
        assert _PIPES["int"]("3.14") == 3

    def test_int_invalid(self) -> None:
        assert _PIPES["int"]("abc") == 0

    @pytest.mark.parametrize("value", ["inf", "-inf", "nan"])
    def test_int_special_values(self, value: str) -> None:
        assert _PIPES["int"](value) == 0

    def test_string_from_list(self) -> None:
        assert _PIPES["string"](["a", "b"]) == "a, b"

    def test_string_from_int(self) -> None:
        assert _PIPES["string"](42) == "42"

    def test_truncate(self) -> None:
        assert _PIPES["truncate"]("hello world", "5") == "hello"

    def test_truncate_short_string(self) -> None:
        assert _PIPES["truncate"]("hi", "100") == "hi"

    def test_truncate_non_integer_length(self) -> None:
        """Non-integer length returns text unchanged."""
        assert _PIPES["truncate"]("hello world", "abc") == "hello world"

    def test_prepend(self) -> None:
        assert _PIPES["prepend"]("world", "hello ") == "hello world"

    def test_append(self) -> None:
        assert _PIPES["append"]("hello", " world") == "hello world"

    def test_match(self) -> None:
        assert _PIPES["match"]("video_12345_hd", r"\d+") == "12345"

    def test_match_no_match(self) -> None:
        assert _PIPES["match"]("no numbers", r"\d+") == ""

    def test_match_invalid_regex(self) -> None:
        assert _PIPES["match"]("test", "[invalid") == ""

    def test_match_redos_pattern_returns_empty(self) -> None:
        """A catastrophic backtracking pattern must not hang — re2 handles this."""
        assert _PIPES["match"]("a" * 30, r"(a+)+b") == ""

    def test_match_empty_pattern(self) -> None:
        assert _PIPES["match"]("test", "") == ""

    def test_pad_already_wide(self) -> None:
        assert _PIPES["pad"]("123", "2") == "123"

    def test_pad_float_truncates(self) -> None:
        """pad converts to int first — 3.7 becomes 3, then zero-padded."""
        assert _PIPES["pad"]("3.7", "2") == "03"

    def test_pad_non_numeric(self) -> None:
        assert _PIPES["pad"]("abc", "2") == "abc"

    def test_json_get_from_string(self) -> None:
        assert (
            _PIPES["json_get"]('{"url": "https://example.com"}', "url")
            == "https://example.com"
        )

    def test_json_get_from_dict(self) -> None:
        assert (
            _PIPES["json_get"]({"url": "https://example.com"}, "url")
            == "https://example.com"
        )

    def test_json_get_missing_key(self) -> None:
        assert _PIPES["json_get"]({"a": 1}, "b") is None

    def test_json_get_invalid_json(self) -> None:
        assert _PIPES["json_get"]("not json", "key") is None


class TestRefEdgeCases:
    """Edge cases for namespace.key splitting, tested through the public API."""

    def test_multi_dot_splits_on_first_dot(self) -> None:
        """slug.nested.key -> namespace='slug', key='nested.key'."""
        lookup = _make_lookup(nodes={"slug": {"nested.key": "found"}})
        assert resolve_expression("{slug.nested.key}", lookup) == "found"

    def test_trailing_dot_raises_at_resolve(self) -> None:
        """slug. has empty key — raises ExpressionError during resolution."""
        lookup = _make_lookup()
        with pytest.raises(ExpressionError, match="key cannot be empty"):
            resolve_expression("{slug.}", lookup)


def _make_lookup(
    nodes: dict[str, dict[str, object]] | None = None,
    run_input: dict[str, object] | None = None,
    config: dict[str, object] | None = None,
) -> Callable[[str, str], object | None]:
    """Build a test lookup function from simple dicts."""
    _nodes = nodes or {}
    _input = run_input or {}
    _config = config or {}

    def lookup(namespace: str, key: str) -> object | None:
        if namespace == "input":
            return _input.get(key)
        if namespace == "config":
            return _config.get(key)
        node_data = _nodes.get(namespace)
        if node_data is None:
            return None
        return node_data.get(key)

    return lookup


class TestResolveExpression:
    def test_single_ref_from_node(self) -> None:
        lookup = _make_lookup(nodes={"export_1": {"url": "https://example.com"}})
        assert resolve_expression("{export_1.url}", lookup) == "https://example.com"

    def test_single_ref_from_input(self) -> None:
        lookup = _make_lookup(run_input={"source": "input://abc"})
        assert resolve_expression("{input.source}", lookup) == "input://abc"

    def test_single_ref_from_config(self) -> None:
        lookup = _make_lookup(config={"quality": "1080p"})
        assert resolve_expression("{config.quality}", lookup) == "1080p"

    def test_single_ref_preserves_list(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"urls": ["a", "b", "c"]}})
        assert resolve_expression("{src_1.urls}", lookup) == ["a", "b", "c"]

    def test_single_ref_preserves_int(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"size": 42000}})
        assert resolve_expression("{src_1.size}", lookup) == 42000

    def test_single_ref_with_pipe(self) -> None:
        lookup = _make_lookup(nodes={"export_1": {"url": "https://example.com"}})
        assert (
            resolve_expression("{export_1.url | upper}", lookup)
            == "HTTPS://EXAMPLE.COM"
        )

    def test_multi_ref_interpolation(self) -> None:
        lookup = _make_lookup(
            nodes={"resolve_1": {"title": "Inception", "year": "2010"}}
        )
        result = resolve_expression("{resolve_1.title} ({resolve_1.year})", lookup)
        assert result == "Inception (2010)"

    def test_missing_ref_no_pipes(self) -> None:
        lookup = _make_lookup()
        assert resolve_expression("{missing.key}", lookup) is None

    def test_missing_ref_with_pipes(self) -> None:
        lookup = _make_lookup()
        assert resolve_expression("{missing.key | upper}", lookup) == ""

    def test_missing_ref_with_default(self) -> None:
        lookup = _make_lookup()
        assert (
            resolve_expression("{missing.key | default(fallback)}", lookup)
            == "fallback"
        )

    def test_missing_ref_required_raises(self) -> None:
        lookup = _make_lookup()
        with pytest.raises(ExpressionError, match="Required"):
            resolve_expression("{missing.key | required}", lookup)

    def test_pipe_chain(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"name": "  Hello World  "}})
        assert (
            resolve_expression("{src_1.name | trim | lower}", lookup) == "hello world"
        )

    def test_path_pipe(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"file_path": "/data/file.txt"}})
        assert resolve_expression("{src_1.file_path | stem}", lookup) == "file"

    def test_mixed_namespaces(self) -> None:
        lookup = _make_lookup(
            nodes={"src_1": {"file_name": "file.txt"}},
            run_input={"prefix": "MY"},
        )
        result = resolve_expression("{input.prefix} - {src_1.file_name | stem}", lookup)
        assert result == "MY - file"

    def test_missing_in_interpolation_becomes_empty(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"name": "hello"}})
        result = resolve_expression("{src_1.name} {missing.key}", lookup)
        assert result == "hello "

    def test_all_missing_interpolation_returns_none(self) -> None:
        lookup = _make_lookup()
        assert resolve_expression("{a.x}{b.y}", lookup) is None

    def test_plain_text_passthrough(self) -> None:
        lookup = _make_lookup()
        assert resolve_expression("just text", lookup) == "just text"

    def test_empty_expression(self) -> None:
        lookup = _make_lookup()
        assert resolve_expression("", lookup) is None

    def test_real_world_url_with_basename(self) -> None:
        lookup = _make_lookup(
            nodes={"export_1": {"url": "https://example.com/v/abc123"}}
        )
        result = resolve_expression("{export_1.url | basename}", lookup)
        assert result == "abc123"

    def test_real_world_filename_transform(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"file_name": "Some.Name.2024.Data.txt"}})
        result = resolve_expression(
            "{src_1.file_name | stem | replace(.,-) | lower}", lookup
        )
        assert result == "some-name-2024-data"

    def test_list_with_join_pipe(self) -> None:
        lookup = _make_lookup(
            nodes={"collect_1": {"urls": ["https://a.com", "https://b.com"]}}
        )
        result = resolve_expression("{collect_1.urls | join(;)}", lookup)
        assert result == "https://a.com;https://b.com"

    def test_nth_pipe_on_list(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"files": ["a.txt", "b.txt"]}})
        assert resolve_expression("{src_1.files | nth(1)}", lookup) == "b.txt"

    def test_escaped_braces_in_expression(self) -> None:
        lookup = _make_lookup(nodes={"src_1": {"name": "test"}})
        result = resolve_expression("\\{literal\\} {src_1.name}", lookup)
        assert result == "{literal} test"

    def test_pipe_arity_mismatch_raises(self) -> None:
        """Pipe called with wrong number of args raises ExpressionError."""
        lookup = _make_lookup(nodes={"src_1": {"name": "hello"}})
        with pytest.raises(ExpressionError, match="replace"):
            resolve_expression("{src_1.name | replace}", lookup)


class TestValidateExpression:
    def test_valid_plain_text(self) -> None:
        assert validate_expression("just text") is None

    def test_valid_multiple_pipes(self) -> None:
        assert validate_expression("{s.k | trim | upper | truncate(50)}") is None

    def test_empty_string(self) -> None:
        assert validate_expression("") is None

    @pytest.mark.parametrize(
        "expr,expected_fragment",
        [
            ("{slug.key", "Unclosed"),
            ("{}", "Empty"),
            ("{nodot}", "namespace.key"),
        ],
    )
    def test_tokenizer_errors_propagate(
        self, expr: str, expected_fragment: str
    ) -> None:
        err = validate_expression(expr)
        assert err is not None
        assert expected_fragment in err


class TestPipeArityValidation:
    """Pipe argument count validation in validate_expression."""

    def test_upper_no_args_ok(self) -> None:
        assert validate_expression("{slug.key | upper}") is None

    def test_upper_with_arg_rejected(self) -> None:
        err = validate_expression("{slug.key | upper(x)}")
        assert err is not None
        assert "expects 0 arg(s)" in err

    def test_replace_one_arg_ok(self) -> None:
        assert validate_expression("{slug.key | replace(a)}") is None

    def test_replace_two_args_ok(self) -> None:
        assert validate_expression("{slug.key | replace(a,b)}") is None

    def test_replace_no_args_rejected(self) -> None:
        err = validate_expression("{slug.key | replace}")
        assert err is not None
        assert "expects 1-2 arg(s)" in err

    def test_unknown_pipe_rejected(self) -> None:
        err = validate_expression("{slug.key | bogus}")
        assert err is not None
        assert "Unknown pipe" in err


class TestExtractRefs:
    def test_node_slugs(self) -> None:
        slugs, keys = extract_refs("{export_1.url} {fetch_1.path}")
        assert slugs == {"export_1", "fetch_1"}
        assert keys == set()

    def test_runtime_keys(self) -> None:
        slugs, keys = extract_refs("{input.source} {input.quality}")
        assert slugs == set()
        assert keys == {"source", "quality"}

    def test_config_excluded(self) -> None:
        slugs, keys = extract_refs("{config.output_dir}")
        assert slugs == set()
        assert keys == set()

    def test_mixed(self) -> None:
        slugs, keys = extract_refs("{export_1.url | upper} - {input.name}")
        assert slugs == {"export_1"}
        assert keys == {"name"}

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(ExpressionError):
            extract_refs("{unclosed")

    def test_plain_text_returns_empty(self) -> None:
        slugs, keys = extract_refs("no references")
        assert slugs == set()
        assert keys == set()

    def test_deduplicates(self) -> None:
        slugs, keys = extract_refs("{src_1.a} {src_1.b}")
        assert slugs == {"src_1"}

    def test_call_token_refs_extracted(self) -> None:
        slugs, keys = extract_refs("{list(export_1.url, export_2.url)}")
        assert slugs == {"export_1", "export_2"}

    def test_call_with_input_ref(self) -> None:
        slugs, keys = extract_refs("{list(input.name, node.val)}")
        assert slugs == {"node"}
        assert keys == {"name"}


class TestListFunction:
    """Tests for the list() function call syntax.

    Resolver and error cases covered by expression_fixtures.json are tested
    in TestSharedFixtures. Only tokenizer and validation tests remain here.
    """

    def test_tokenize_list_basic(self) -> None:
        tokens = tokenize_expression("{list(a.x, b.y)}")
        assert len(tokens) == 1
        assert tokens[0].kind == "call"
        assert tokens[0].value == "list"
        assert tokens[0].refs == ("a.x", "b.y")
        assert tokens[0].pipes == ()

    def test_tokenize_list_with_pipes(self) -> None:
        tokens = tokenize_expression("{list(a.x, b.y) | join(-)}")
        assert len(tokens) == 1
        assert tokens[0].kind == "call"
        assert tokens[0].refs == ("a.x", "b.y")
        assert tokens[0].pipes == (PipeCall("join", ("-",)),)

    def test_tokenize_list_empty(self) -> None:
        tokens = tokenize_expression("{list()}")
        assert len(tokens) == 1
        assert tokens[0].kind == "call"
        assert tokens[0].refs == ()

    def test_tokenize_list_single_arg(self) -> None:
        tokens = tokenize_expression("{list(a.x)}")
        assert tokens[0].refs == ("a.x",)

    def test_validate_list_valid(self) -> None:
        assert validate_expression("{list(a.x, b.y)}") is None

    def test_validate_list_with_pipes(self) -> None:
        assert validate_expression("{list(a.x) | join(-)}") is None

    def test_validate_list_unknown_pipe(self) -> None:
        assert validate_expression("{list(a.x) | bogus}") == "Unknown pipe 'bogus'"


class TestFlattenPipe:
    def test_flatten_nested(self) -> None:
        fn = _PIPES["flatten"]
        assert fn([[1, 2], 3, [4]]) == [1, 2, 3, 4]

    def test_flatten_already_flat(self) -> None:
        fn = _PIPES["flatten"]
        assert fn([1, 2, 3]) == [1, 2, 3]

    def test_flatten_non_list(self) -> None:
        fn = _PIPES["flatten"]
        assert fn("hello") == "hello"

    def test_flatten_one_level_only(self) -> None:
        """Non-recursive: only one level of nesting removed."""
        fn = _PIPES["flatten"]
        assert fn([[[1]], [2]]) == [[1], 2]


class TestCompactPipe:
    def test_compact_removes_none(self) -> None:
        fn = _PIPES["compact"]
        assert fn([None, "a", None, "b"]) == ["a", "b"]

    def test_compact_preserves_empty_string(self) -> None:
        """compact strips only None, not empty strings or falsy values."""
        fn = _PIPES["compact"]
        assert fn([None, "", 0, False, "a"]) == ["", 0, False, "a"]

    def test_compact_non_list(self) -> None:
        fn = _PIPES["compact"]
        assert fn("hello") == "hello"


def _fixture_lookup(data: dict) -> Callable[[str, str], object | None]:
    return lambda ns, key: data.get(f"{ns}.{key}")


_RESOLVER_FIXTURES = [
    ("{list(a.x, b.y)}", {"a.x": "1", "b.y": "2"}, ["1", "2"]),
    ("{list(a.x, b.y) | join(-)}", {"a.x": "1", "b.y": "2"}, "1-2"),
    ("{list(a.x, b.y) | first}", {"a.x": "1", "b.y": "2"}, "1"),
    ("{list(a.missing, b.y)}", {"b.y": "2"}, [None, "2"]),
    ("{list(a.missing, b.y) | compact}", {"b.y": "2"}, ["2"]),
    ("{list()}", {}, []),
    ("{list(a.x)}", {"a.x": "1"}, ["1"]),
    ("prefix {list(a.x, b.y)} suffix", {"a.x": "1", "b.y": "2"}, "prefix 1, 2 suffix"),
    ("{a.val | pad(2)}", {"a.val": "3"}, "03"),
    ("{a.val | pad(3)}", {"a.val": "3"}, "003"),
    ("{list(a.x, b.y) | compact | first}", {"b.y": "2"}, "2"),
    ("{list(a.x, b.y) | flatten}", {"a.x": ["1", "2"], "b.y": "3"}, ["1", "2", "3"]),
]

_ERROR_FIXTURES = [
    ("{list(bad)}", "Invalid reference 'bad' at position 0: expected 'namespace.key'"),
    ("{unknown_fn(a.b)}", "Unknown function 'unknown_fn'"),
]


@pytest.mark.parametrize(
    "expression,lookup_data,expected",
    _RESOLVER_FIXTURES,
    ids=[f[0] for f in _RESOLVER_FIXTURES],
)
def test_resolver_fixture(expression: str, lookup_data: dict, expected: object) -> None:
    tokenize_expression.cache_clear()
    lookup = _fixture_lookup(lookup_data)
    result = resolve_expression(expression, lookup)
    assert result == expected


@pytest.mark.parametrize(
    "expression,expected_error",
    _ERROR_FIXTURES,
    ids=[f[0] for f in _ERROR_FIXTURES],
)
def test_error_fixture(expression: str, expected_error: str) -> None:
    tokenize_expression.cache_clear()
    error = validate_expression(expression)
    if error is None:
        try:
            tokenize_expression(expression)
            pytest.fail(
                f"Expression {expression!r} should have errored: {expected_error}"
            )
        except ExpressionError as e:
            assert expected_error in str(e)
    else:
        assert expected_error in error
