from tiv_tts.text import CharacterTokenizer, normalize_text


def test_normalization_preserves_tiv_diacritics() -> None:
    result = normalize_text("  Tsô\u00a0ka—hemen  ")
    assert result.cleaned == "Tsô ka-hemen"
    assert "ô" in result.cleaned


def test_character_tokenizer_round_trip() -> None:
    text = "Aôndo yô"
    tokenizer = CharacterTokenizer.from_texts([text])
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_uncertain_scripts_are_flagged() -> None:
    result = normalize_text("Kра kwagh")
    assert "non_latin_letter" in result.warnings

