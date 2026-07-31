# Tiv TTS dataset audit

Generated: 2026-07-30T06:10:37.994314+00:00

## CURRENT STATE

- Repository: No existing source code, Git repository, README, configuration, or dependency manifest was present at inspection time. The audit script is the first project code added.
- Dataset: `Tiv-TTS-Dataset/ (repository root)` contains 14 source-group exports.
- Framework: No TTS framework or pretrained model has been selected or installed.
- Environment: Python 3.14.3; audio metadata probed with `afinfo`. PyTorch, torchaudio, FFmpeg, and common Python audio packages were not present at inspection time.
- Risks: Speaker/session identity is unverified; licence and consent are unverified; waveform-level quality remains unassessed; the directory is not currently a Git repository.

## DATASET AUDIT

- Files: 2,443 mapped clips; 2,443 total sample records including unmatched files.
- Duration: 5.8658 hours (21,116.880 seconds).
- Export-name duration: 7.4717 hours; this is not used as selected-audio duration because every source-group name overstates the sum of its final MP3 files.
- Speakers: Unknown. No explicit speaker, gender, age, dialect, or region metadata is present.
- Source groups: 14 export directories; do not treat these as speakers until provenance is confirmed.
- Audio formats: codecs `{"mp3": 2443}`; containers `{"MPG3": 2443}`.
- Sample rates: `{"48000": 2443}`.
- Channels: `{"1": 2443}`.
- Bit depth: Not available for compressed MP3 without decoding.
- Transcript format: UTF-8 TSV with columns `audio_filename, key, sentence, attempts`.
- Invalid samples: 0.
- Warning samples: 113.
- Valid samples: 2,330.
- Recommended exclusions: 0; only objective mapping/audio-integrity failures and clips below the configured minimum are automatically recommended.
- Duration statistics: minimum 2.136 s; maximum 31.776 s; mean 8.644 s; median 8.304 s; p05 3.936 s; p95 14.136 s.
- Audio without transcripts: 0.
- Transcripts without audio: 0.
- Duplicate audio: 0 exact SHA-256 groups.
- Duplicate transcripts: 0 exact-text groups; these are warnings, not automatic exclusions.
- Empty transcripts: 0.
- Unicode NFC inconsistencies: 0.
- Character inventory: 130 unique characters including whitespace. Full inventory: ' ' (U+0020, SPACE, 31947); '!' (U+0021, EXCLAMATION MARK, 25); '"' (U+0022, QUOTATION MARK, 92); '#' (U+0023, NUMBER SIGN, 3); "'" (U+0027, APOSTROPHE, 77); '(' (U+0028, LEFT PARENTHESIS, 17); ')' (U+0029, RIGHT PARENTHESIS, 16); ',' (U+002C, COMMA, 685); '-' (U+002D, HYPHEN-MINUS, 237); '.' (U+002E, FULL STOP, 1809); '/' (U+002F, SOLIDUS, 8); '0' (U+0030, DIGIT ZERO, 28); '1' (U+0031, DIGIT ONE, 81); '2' (U+0032, DIGIT TWO, 27); '3' (U+0033, DIGIT THREE, 15); '4' (U+0034, DIGIT FOUR, 16); '5' (U+0035, DIGIT FIVE, 16); '6' (U+0036, DIGIT SIX, 9); '7' (U+0037, DIGIT SEVEN, 12); '8' (U+0038, DIGIT EIGHT, 22); '9' (U+0039, DIGIT NINE, 38); ':' (U+003A, COLON, 24); ';' (U+003B, SEMICOLON, 80); '?' (U+003F, QUESTION MARK, 35); 'A' (U+0041, LATIN CAPITAL LETTER A, 636); 'B' (U+0042, LATIN CAPITAL LETTER B, 54); 'C' (U+0043, LATIN CAPITAL LETTER C, 20); 'D' (U+0044, LATIN CAPITAL LETTER D, 55); 'E' (U+0045, LATIN CAPITAL LETTER E, 140); 'F' (U+0046, LATIN CAPITAL LETTER F, 16); 'G' (U+0047, LATIN CAPITAL LETTER G, 148); 'H' (U+0048, LATIN CAPITAL LETTER H, 111); 'I' (U+0049, LATIN CAPITAL LETTER I, 193); 'J' (U+004A, LATIN CAPITAL LETTER J, 21); 'K' (U+004B, LATIN CAPITAL LETTER K, 423); 'L' (U+004C, LATIN CAPITAL LETTER L, 91); 'M' (U+004D, LATIN CAPITAL LETTER M, 187); 'N' (U+004E, LATIN CAPITAL LETTER N, 334); 'O' (U+004F, LATIN CAPITAL LETTER O, 76); 'P' (U+0050, LATIN CAPITAL LETTER P, 27); 'R' (U+0052, LATIN CAPITAL LETTER R, 29); 'S' (U+0053, LATIN CAPITAL LETTER S, 266); 'T' (U+0054, LATIN CAPITAL LETTER T, 490); 'U' (U+0055, LATIN CAPITAL LETTER U, 94); 'V' (U+0056, LATIN CAPITAL LETTER V, 94); 'W' (U+0057, LATIN CAPITAL LETTER W, 286); 'Y' (U+0059, LATIN CAPITAL LETTER Y, 45); 'Z' (U+005A, LATIN CAPITAL LETTER Z, 20); '\\' (U+005C, REVERSE SOLIDUS, 134); '_' (U+005F, LOW LINE, 1); 'a' (U+0061, LATIN SMALL LETTER A, 19102); 'b' (U+0062, LATIN SMALL LETTER B, 1804); 'c' (U+0063, LATIN SMALL LETTER C, 465); 'd' (U+0064, LATIN SMALL LETTER D, 1845); 'e' (U+0065, LATIN SMALL LETTER E, 12115); 'f' (U+0066, LATIN SMALL LETTER F, 355); 'g' (U+0067, LATIN SMALL LETTER G, 4334); 'h' (U+0068, LATIN SMALL LETTER H, 5195); 'i' (U+0069, LATIN SMALL LETTER I, 6728); 'j' (U+006A, LATIN SMALL LETTER J, 429); 'k' (U+006B, LATIN SMALL LETTER K, 4459); 'l' (U+006C, LATIN SMALL LETTER L, 2961); 'm' (U+006D, LATIN SMALL LETTER M, 3978); 'n' (U+006E, LATIN SMALL LETTER N, 12079); 'o' (U+006F, LATIN SMALL LETTER O, 3825); 'p' (U+0070, LATIN SMALL LETTER P, 1279); 'r' (U+0072, LATIN SMALL LETTER R, 5058); 's' (U+0073, LATIN SMALL LETTER S, 3852); 't' (U+0074, LATIN SMALL LETTER T, 1961); 'u' (U+0075, LATIN SMALL LETTER U, 6781); 'v' (U+0076, LATIN SMALL LETTER V, 3125); 'w' (U+0077, LATIN SMALL LETTER W, 1924); 'x' (U+0078, LATIN SMALL LETTER X, 1); 'y' (U+0079, LATIN SMALL LETTER Y, 2895); 'z' (U+007A, LATIN SMALL LETTER Z, 684); 'Ä' (U+00C4, LATIN CAPITAL LETTER A WITH DIAERESIS, 1); 'Í' (U+00CD, LATIN CAPITAL LETTER I WITH ACUTE, 1); 'Ô' (U+00D4, LATIN CAPITAL LETTER O WITH CIRCUMFLEX, 15); 'Õ' (U+00D5, LATIN CAPITAL LETTER O WITH TILDE, 3); 'Ö' (U+00D6, LATIN CAPITAL LETTER O WITH DIAERESIS, 1); 'Ú' (U+00DA, LATIN CAPITAL LETTER U WITH ACUTE, 3); 'à' (U+00E0, LATIN SMALL LETTER A WITH GRAVE, 2); 'á' (U+00E1, LATIN SMALL LETTER A WITH ACUTE, 23); 'â' (U+00E2, LATIN SMALL LETTER A WITH CIRCUMFLEX, 1); 'å' (U+00E5, LATIN SMALL LETTER A WITH RING ABOVE, 2); 'é' (U+00E9, LATIN SMALL LETTER E WITH ACUTE, 33); 'ê' (U+00EA, LATIN SMALL LETTER E WITH CIRCUMFLEX, 1); 'í' (U+00ED, LATIN SMALL LETTER I WITH ACUTE, 9); 'î' (U+00EE, LATIN SMALL LETTER I WITH CIRCUMFLEX, 3); 'ò' (U+00F2, LATIN SMALL LETTER O WITH GRAVE, 2); 'ó' (U+00F3, LATIN SMALL LETTER O WITH ACUTE, 26); 'ô' (U+00F4, LATIN SMALL LETTER O WITH CIRCUMFLEX, 1687); 'õ' (U+00F5, LATIN SMALL LETTER O WITH TILDE, 35); 'ö' (U+00F6, LATIN SMALL LETTER O WITH DIAERESIS, 26); 'ú' (U+00FA, LATIN SMALL LETTER U WITH ACUTE, 31); 'û' (U+00FB, LATIN SMALL LETTER U WITH CIRCUMFLEX, 9); 'ą' (U+0105, LATIN SMALL LETTER A WITH OGONEK, 1); 'ĉ' (U+0109, LATIN SMALL LETTER C WITH CIRCUMFLEX, 1); 'ę' (U+0119, LATIN SMALL LETTER E WITH OGONEK, 1); 'ı' (U+0131, LATIN SMALL LETTER DOTLESS I, 3); 'ļ' (U+013C, LATIN SMALL LETTER L WITH CEDILLA, 1); 'ņ' (U+0146, LATIN SMALL LETTER N WITH CEDILLA, 1); 'ŋ' (U+014B, LATIN SMALL LETTER ENG, 1); 'ō' (U+014D, LATIN SMALL LETTER O WITH MACRON, 64); 'ő' (U+0151, LATIN SMALL LETTER O WITH DOUBLE ACUTE, 4); 'š' (U+0161, LATIN SMALL LETTER S WITH CARON, 1); 'ų' (U+0173, LATIN SMALL LETTER U WITH OGONEK, 1); 'ſ' (U+017F, LATIN SMALL LETTER LONG S, 2); '̠' (U+0320, COMBINING MINUS SIGN BELOW, 1); 'Α' (U+0391, GREEK CAPITAL LETTER ALPHA, 1); 'Κ' (U+039A, GREEK CAPITAL LETTER KAPPA, 1); 'ο' (U+03BF, GREEK SMALL LETTER OMICRON, 1); 'τ' (U+03C4, GREEK SMALL LETTER TAU, 2); 'υ' (U+03C5, GREEK SMALL LETTER UPSILON, 1); 'ό' (U+03CC, GREEK SMALL LETTER OMICRON WITH TONOS, 1); 'К' (U+041A, CYRILLIC CAPITAL LETTER KA, 1); 'а' (U+0430, CYRILLIC SMALL LETTER A, 2); 'е' (U+0435, CYRILLIC SMALL LETTER IE, 3); 'к' (U+043A, CYRILLIC SMALL LETTER KA, 1); 'о' (U+043E, CYRILLIC SMALL LETTER O, 2); 'п' (U+043F, CYRILLIC SMALL LETTER PE, 3); 'р' (U+0440, CYRILLIC SMALL LETTER ER, 4); 'т' (U+0442, CYRILLIC SMALL LETTER TE, 2); 'ו' (U+05D5, HEBREW LETTER VAV, 1); 'ח' (U+05D7, HEBREW LETTER HET, 1); 'י' (U+05D9, HEBREW LETTER YOD, 1); 'פ' (U+05E4, HEBREW LETTER PE, 1); 'ố' (U+1ED1, LATIN SMALL LETTER O WITH CIRCUMFLEX AND ACUTE, 2); 'ồ' (U+1ED3, LATIN SMALL LETTER O WITH CIRCUMFLEX AND GRAVE, 1); '🙏' (U+1F64F, PERSON WITH FOLDED HANDS, 1)
- Rare characters (fewer than 5 occurrences): '#' (U+0023, 3); '_' (U+005F, 1); 'x' (U+0078, 1); 'Ä' (U+00C4, 1); 'Í' (U+00CD, 1); 'Õ' (U+00D5, 3); 'Ö' (U+00D6, 1); 'Ú' (U+00DA, 3); 'à' (U+00E0, 2); 'â' (U+00E2, 1); 'å' (U+00E5, 2); 'ê' (U+00EA, 1); 'î' (U+00EE, 3); 'ò' (U+00F2, 2); 'ą' (U+0105, 1); 'ĉ' (U+0109, 1); 'ę' (U+0119, 1); 'ı' (U+0131, 3); 'ļ' (U+013C, 1); 'ņ' (U+0146, 1); 'ŋ' (U+014B, 1); 'ő' (U+0151, 4); 'š' (U+0161, 1); 'ų' (U+0173, 1); 'ſ' (U+017F, 2); '̠' (U+0320, 1); 'Α' (U+0391, 1); 'Κ' (U+039A, 1); 'ο' (U+03BF, 1); 'τ' (U+03C4, 2); 'υ' (U+03C5, 1); 'ό' (U+03CC, 1); 'К' (U+041A, 1); 'а' (U+0430, 2); 'е' (U+0435, 3); 'к' (U+043A, 1); 'о' (U+043E, 2); 'п' (U+043F, 3); 'р' (U+0440, 4); 'т' (U+0442, 2); 'ו' (U+05D5, 1); 'ח' (U+05D7, 1); 'י' (U+05D9, 1); 'פ' (U+05E4, 1); 'ố' (U+1ED1, 2); 'ồ' (U+1ED3, 1); '🙏' (U+1F64F, 1)
- Visually similar character groups: `[{"normalization_key": "HOMOGLYPH:A", "characters": ["A", "Α"], "codepoints": ["U+0041", "U+0391"]}, {"normalization_key": "HOMOGLYPH:K", "characters": ["K", "Κ", "К"], "codepoints": ["U+004B", "U+039A", "U+041A"]}, {"normalization_key": "HOMOGLYPH:a", "characters": ["a", "а"], "codepoints": ["U+0061", "U+0430"]}, {"normalization_key": "HOMOGLYPH:e", "characters": ["e", "е"], "codepoints": ["U+0065", "U+0435"]}, {"normalization_key": "HOMOGLYPH:k", "characters": ["k", "к"], "codepoints": ["U+006B", "U+043A"]}, {"normalization_key": "HOMOGLYPH:o", "characters": ["o", "ο", "о"], "codepoints": ["U+006F", "U+03BF", "U+043E"]}, {"normalization_key": "HOMOGLYPH:p", "characters": ["p", "р"], "codepoints": ["U+0070", "U+0440"]}, {"normalization_key": "HOMOGLYPH:t", "characters": ["t", "τ", "т"], "codepoints": ["U+0074", "U+03C4", "U+0442"]}, {"normalization_key": "HOMOGLYPH:u", "characters": ["u", "υ"], "codepoints": ["U+0075", "U+03C5"]}, {"normalization_key": "s", "characters": ["s", "ſ"], "codepoints": ["U+0073", "U+017F"]}]`.
- Rows containing digits: 80.
- Rows containing all-capital tokens: 31.
- Rows containing literal backslash escapes: 95; these require conservative quote-normalisation review.
- Rows containing non-Latin letters: 8; Greek, Cyrillic, and Hebrew fragments require manual review.
- Rows containing emoji or other standalone symbols: 1.
- Recording-attempt distribution: `{"1": 1978, "2": 401, "3": 51, "4": 8, "42": 1, "5": 3, "6": 1}`; values above 10 are flagged for review.
- Mixed English/Tiv: Not automatically classified: Tiv and English share Latin-script vocabulary patterns; native-speaker or language-ID review is required.
- Licensing/consent status: Unverified: no licence, consent, release, or provenance file was found in the dataset root.

### Source-group totals

| Source group | Audio | TSV rows | Readable | Named s | Actual s | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| tts_Tiv_dataset_10_175clips_2201s_20260415-1732 | 175 | 175 | 175 | 2201 | 1769.736 | 80.4% |
| tts_Tiv_dataset_11_174clips_2211s_20260415-1859 | 174 | 174 | 174 | 2211 | 1629.768 | 73.7% |
| tts_Tiv_dataset_12_174clips_2319s_20260415-2010 | 174 | 174 | 174 | 2319 | 1712.160 | 73.8% |
| tts_Tiv_dataset_13_175clips_1775s_20260417-1541 | 175 | 175 | 175 | 1775 | 1515.696 | 85.4% |
| tts_Tiv_dataset_14_174clips_1801s_20260417-1719 | 174 | 174 | 174 | 1801 | 1565.832 | 86.9% |
| tts_Tiv_dataset_15_175clips_2452s_20260418-0814 | 175 | 175 | 175 | 2452 | 1408.968 | 57.5% |
| tts_Tiv_dataset_16_174clips_1819s_20260418-0917 | 174 | 174 | 174 | 1819 | 1400.664 | 77.0% |
| tts_Tiv_dataset_18_175clips_2077s_20260418-1037 | 175 | 175 | 175 | 2077 | 1402.056 | 67.5% |
| tts_Tiv_dataset_19_175clips_1863s_20260418-1133 | 175 | 175 | 175 | 1863 | 1456.680 | 78.2% |
| tts_Tiv_dataset_20_175clips_1866s_20260418-1226 | 175 | 175 | 175 | 1866 | 1372.896 | 73.6% |
| tts_Tiv_dataset_4_172clips_1381s_20260413-2305 | 172 | 172 | 172 | 1381 | 1279.848 | 92.7% |
| tts_Tiv_dataset_6_175clips_1648s_20260414-1529 | 175 | 175 | 175 | 1648 | 1548.960 | 94.0% |
| tts_Tiv_dataset_7_175clips_1580s_20260414-1944 | 175 | 175 | 175 | 1580 | 1447.416 | 91.6% |
| tts_Tiv_dataset_9_175clips_1905s_20260415-1550 | 175 | 175 | 175 | 1905 | 1606.200 | 84.3% |

### Dataset-level warnings

- tts_Tiv_dataset_10_175clips_2201s_20260415-1732: directory name declares 2201 seconds, while selected MP3 files total 1769.736 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_11_174clips_2211s_20260415-1859: directory name declares 2211 seconds, while selected MP3 files total 1629.768 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_12_174clips_2319s_20260415-2010: directory name declares 2319 seconds, while selected MP3 files total 1712.160 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_13_175clips_1775s_20260417-1541: directory name declares 1775 seconds, while selected MP3 files total 1515.696 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_14_174clips_1801s_20260417-1719: directory name declares 1801 seconds, while selected MP3 files total 1565.832 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_15_175clips_2452s_20260418-0814: directory name declares 2452 seconds, while selected MP3 files total 1408.968 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_16_174clips_1819s_20260418-0917: directory name declares 1819 seconds, while selected MP3 files total 1400.664 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_18_175clips_2077s_20260418-1037: directory name declares 2077 seconds, while selected MP3 files total 1402.056 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_19_175clips_1863s_20260418-1133: directory name declares 1863 seconds, while selected MP3 files total 1456.680 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_20_175clips_1866s_20260418-1226: directory name declares 1866 seconds, while selected MP3 files total 1372.896 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_4_172clips_1381s_20260413-2305: directory name declares 1381 seconds, while selected MP3 files total 1279.848 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_6_175clips_1648s_20260414-1529: directory name declares 1648 seconds, while selected MP3 files total 1548.960 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_7_175clips_1580s_20260414-1944: directory name declares 1580 seconds, while selected MP3 files total 1447.416 seconds; the declared value may include discarded attempts or session time
- tts_Tiv_dataset_9_175clips_1905s_20260415-1550: directory name declares 1905 seconds, while selected MP3 files total 1606.200 seconds; the declared value may include discarded attempts or session time

### Recommended exclusions

- None from metadata and mapping checks.

## AUDIT LIMITATIONS

- Signal-level checks are `not_assessed`: long_silences, excessive_background_noise, clipping, very_low_volume.
- Reason: No supported signal-analysis decoder is installed. Metadata probing cannot establish waveform-level quality.
- Background-language classification and transcript correctness require native-Tiv review.
- The 14 source groups cannot safely be used as speaker IDs or session IDs until their provenance is confirmed.

## RECOMMENDED NEXT STEP

- Action: Confirm speaker/session metadata, contributor consent, dataset licence, and whether the 14 export directories belong to one or multiple speakers; then add a pinned local audio-analysis dependency and run the waveform-quality extension of this audit.
- Reason: Those facts determine leakage-safe splits, model architecture, release eligibility, and which samples need manual review.
- Expected output: Verified provenance metadata plus a complete audit covering silence, noise, clipping, loudness, and a reviewed exclusion list.
- Command to reproduce this report: `python3 scripts/audit_dataset.py --dataset Tiv-TTS-Dataset`

Machine-readable details are in `outputs/dataset_audit/audit.json` and the per-sample classifications are in `outputs/dataset_audit/samples.csv`.
