"""
Token grammars for number/date/datetime fields.

A grammar answers "given the text decoded so far, which symbols may come next?" Every
symbol must be exactly one token, so the engine can softmax over the allowed symbols'
logits at each step. `text` is the concatenation of chosen symbols, excluding END.
"""

import calendar
from typing import List, Optional, Union

NULL = " null"
END = ",\n"
QUOTE = ' "'
MINUS = " -"  # Qwen encodes '": -5' as '":', ' -', '5'
DIGITS = [str(d) for d in range(10)]
SYMBOLS = DIGITS + ["-", ".", "T", ":", QUOTE, " ", MINUS, NULL, END]

MAX_INT_DIGITS = 15


def build_symbol_vocab(tokenizer) -> dict:
    vocab = {}
    for s in SYMBOLS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"symbol {s!r} encodes to {len(ids)} tokens; typed fields need a tokenizer "
                f"that encodes each of {SYMBOLS!r} as a single token"
            )
        vocab[s] = ids[0]
    return vocab


def _digits(lo: int, hi: int) -> List[str]:
    return DIGITS[lo:hi + 1]


class NumberGrammar:
    """Emits ' ' then a number such as 12 or 3.5, ' -' then a negative one, or NULL."""

    def __init__(self, decimals: int):
        self.decimals = decimals

    def allowed(self, text: str) -> List[str]:
        if text == "":
            return [NULL, " ", MINUS]
        body = text[1:]
        int_part, dot, frac = body.lstrip("-").partition(".")
        if int_part == "":
            return DIGITS
        if dot:
            if frac == "":
                return DIGITS
            return DIGITS + [END] if len(frac) < self.decimals else [END]
        more = [] if int_part == "0" or len(int_part) >= MAX_INT_DIGITS else DIGITS
        return more + (["."] if self.decimals > 0 else []) + [END]

    def parse(self, text: str) -> Union[int, float]:
        body = text.strip()
        return float(body) if self.decimals > 0 else int(body)


class DateGrammar:
    """Emits ' "' then YYYY-MM-DD (or YYYY-MM-DDTHH:MM:SS), or NULL. No closing quote."""

    LITERALS = {4: "-", 7: "-", 10: "T", 13: ":", 16: ":"}

    def __init__(self, with_time: bool):
        self.length = 19 if with_time else 10

    def allowed(self, text: str) -> List[str]:
        if text == "":
            return [NULL, QUOTE]
        body = text[len(QUOTE):]
        pos = len(body)
        if pos >= self.length:
            return []
        if pos in self.LITERALS:
            return [self.LITERALS[pos]]
        if pos == 0:
            return _digits(1, 2)
        if pos in (1, 2, 3):
            return DIGITS
        if pos == 5:
            return _digits(0, 1)
        if pos == 6:
            return _digits(1, 9) if body[5] == "0" else _digits(0, 2)
        max_day = calendar.monthrange(int(body[0:4]), int(body[5:7]))[1]
        if pos == 8:
            return _digits(0, max_day // 10)
        if pos == 9:
            d1 = int(body[8])
            if d1 == 0:
                return _digits(1, 9)
            if d1 == max_day // 10:
                return _digits(0, max_day % 10)
            return DIGITS
        if pos == 11:
            return _digits(0, 2)
        if pos == 12:
            return _digits(0, 3) if body[11] == "2" else DIGITS
        if pos in (14, 17):
            return _digits(0, 5)
        return DIGITS

    def parse(self, text: str) -> str:
        return text[len(QUOTE):]


def make_grammar(fdef) -> Union[NumberGrammar, DateGrammar]:
    if fdef.field_type == "number":
        return NumberGrammar(fdef.decimals)
    return DateGrammar(with_time=fdef.field_type == "datetime")


class GrammarRow:
    """
    One field's row in lockstep_decode: picks the most likely allowed symbol each step.

    Once finished, `value` holds the parsed value (None if NULL won), `confidence` the
    product of the chosen symbols' probabilities and `p_null` the first step's NULL
    probability.
    """

    def __init__(self, grammar, vocab: dict, temperature: float):
        self.grammar = grammar
        self.vocab = vocab
        self.temperature = max(temperature, 1e-4)
        self.text = ""
        self.value = None
        self.confidence = 1.0
        self.p_null = 0.0

    def step(self, logits) -> Optional[int]:
        import mlx.core as mx

        syms = self.grammar.allowed(self.text)
        ids = mx.array([self.vocab[s] for s in syms])
        probs = mx.softmax(logits[ids].astype(mx.float32) / self.temperature).tolist()
        best = max(range(len(syms)), key=probs.__getitem__)
        s = syms[best]
        self.confidence *= probs[best]
        if self.text == "":
            self.p_null = probs[syms.index(NULL)]
        if s == NULL:
            self.value = None
            return None
        if s == END:
            self.value = self.grammar.parse(self.text)
            return None
        self.text += s
        if self.grammar.allowed(self.text) == []:
            self.value = self.grammar.parse(self.text)
            return None
        return self.vocab[s]
