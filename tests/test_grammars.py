import pytest

from core.grammars import END, NULL, SYMBOLS, DateGrammar, NumberGrammar, build_symbol_vocab

D = set("0123456789")
Q = ' "'


def allowed(g, text):
    return set(g.allowed(text))


def test_number_integer():
    g = NumberGrammar(0)
    assert allowed(g, "") == {NULL, " ", " -"}
    assert allowed(g, " ") == D
    assert allowed(g, " -") == D
    assert allowed(g, " -1") == D | {END}
    assert allowed(g, " 0") == {END}
    assert allowed(g, " 12") == D | {END}
    assert allowed(g, " " + "1" * 15) == {END}
    assert g.parse(" -42") == -42 and isinstance(g.parse(" -42"), int)


def test_number_decimals():
    g = NumberGrammar(2)
    assert allowed(g, " 12") == D | {".", END}
    assert allowed(g, " 1.") == D
    assert allowed(g, " 1.2") == D | {END}
    assert allowed(g, " 1.23") == {END}
    assert g.parse(" 1.5") == 1.5
    assert g.parse(" 3") == 3.0 and isinstance(g.parse(" 3"), float)


def test_date():
    g = DateGrammar(with_time=False)
    assert allowed(g, "") == {NULL, Q}
    assert allowed(g, Q) == {"1", "2"}
    assert allowed(g, Q + "2024") == {"-"}
    assert allowed(g, Q + "2024-") == {"0", "1"}
    assert allowed(g, Q + "2024-0") == D - {"0"}
    assert allowed(g, Q + "2024-1") == {"0", "1", "2"}
    assert allowed(g, Q + "2024-02-") == {"0", "1", "2"}
    assert allowed(g, Q + "2024-02-0") == D - {"0"}
    assert allowed(g, Q + "2024-02-2") == D
    assert allowed(g, Q + "2023-02-2") == D - {"9"}
    assert allowed(g, Q + "2024-04-3") == {"0"}
    assert allowed(g, Q + "2024-01-3") == {"0", "1"}
    assert g.allowed(Q + "2024-10-20") == []
    assert g.parse(Q + "2024-10-20") == "2024-10-20"


def test_datetime():
    g = DateGrammar(with_time=True)
    assert allowed(g, Q + "2024-10-20") == {"T"}
    assert allowed(g, Q + "2024-10-20T") == {"0", "1", "2"}
    assert allowed(g, Q + "2024-10-20T2") == {"0", "1", "2", "3"}
    assert allowed(g, Q + "2024-10-20T23:") == set("012345")
    assert g.allowed(Q + "2024-10-20T23:59:59") == []
    assert g.parse(Q + "2024-10-20T23:59:59") == "2024-10-20T23:59:59"


def test_symbol_vocab_qwen():
    from transformers import AutoTokenizer
    from core.engine_mlx import MODEL_ID

    vocab = build_symbol_vocab(AutoTokenizer.from_pretrained(MODEL_ID))
    assert set(vocab) == set(SYMBOLS)


def test_symbol_vocab_rejects_multi_token_symbol():
    class FakeTokenizer:
        def encode(self, s, add_special_tokens=False):
            return [1, 2] if s == "5" else [ord(s[0])]

    with pytest.raises(ValueError, match="'5'"):
        build_symbol_vocab(FakeTokenizer())


def test_negative_zero_parses_as_zero():
    import math

    assert str(NumberGrammar(0).parse(" -0")) == "0"
    assert math.copysign(1, NumberGrammar(2).parse(" -0.00")) == 1
