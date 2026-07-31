"""Conservative Tiv transcript normalization and character tokenization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


SPACE_TRANSLATION = {
    ord("\u00a0"): " ",
    ord("\u2007"): " ",
    ord("\u202f"): " ",
}
PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """A normalized transcript with an auditable list of transformations."""

    original: str
    cleaned: str
    changes: tuple[str, ...]
    warnings: tuple[str, ...]


def normalize_text(text: str) -> NormalizedText:
    """Normalize safe typography while preserving Tiv letters and diacritics."""

    changes: list[str] = []
    warnings: list[str] = []
    cleaned = unicodedata.normalize("NFC", text)
    if cleaned != text:
        changes.append("unicode_nfc")

    translated = cleaned.translate(SPACE_TRANSLATION)
    if translated != cleaned:
        changes.append("unusual_spaces")
    cleaned = translated

    translated = cleaned.translate(PUNCTUATION_TRANSLATION)
    if translated != cleaned:
        changes.append("typographic_punctuation")
    cleaned = translated

    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    if collapsed != cleaned:
        changes.append("whitespace")
    cleaned = collapsed

    if "\\" in cleaned:
        warnings.append("literal_backslash")
    if any(
        unicodedata.category(char).startswith("L")
        and "LATIN" not in unicodedata.name(char, "")
        for char in cleaned
    ):
        warnings.append("non_latin_letter")
    if any(unicodedata.category(char) == "So" for char in cleaned):
        warnings.append("symbol_or_emoji")

    return NormalizedText(
        original=text,
        cleaned=cleaned,
        changes=tuple(changes),
        warnings=tuple(warnings),
    )


class CharacterTokenizer:
    """A deterministic character tokenizer that preserves corpus diacritics."""

    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocabulary: list[str]) -> None:
        if vocabulary[:2] != [self.PAD, self.UNK]:
            raise ValueError("Vocabulary must start with <pad>, <unk>.")
        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError("Vocabulary entries must be unique.")
        self.vocabulary = vocabulary
        self.char_to_id = {char: index for index, char in enumerate(vocabulary)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "CharacterTokenizer":
        """Build a tokenizer from normalized training texts."""

        characters = sorted(set("".join(texts)), key=ord)
        return cls([cls.PAD, cls.UNK, *characters])

    def encode(self, text: str) -> list[int]:
        """Encode text, retaining unknown characters as explicit UNK tokens."""

        unk = self.char_to_id[self.UNK]
        return [self.char_to_id.get(char, unk) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs for debugging."""

        return "".join(
            self.vocabulary[token_id]
            for token_id in token_ids
            if self.vocabulary[token_id] not in {self.PAD, self.UNK}
        )

