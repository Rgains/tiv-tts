# Tiv TTS signal-quality audit

Generated: 2026-07-30T09:51:12.426663+00:00

- Files analyzed: 2,443
- Readable: 2,443
- Unreadable: 0
- Files with heuristic warnings: 1,215
- Warning counts: `{"high_silence_fraction": 1202, "long_leading_silence": 16, "long_trailing_silence": 155, "possible_clipping": 8, "very_low_volume": 1}`
- Peak dBFS statistics: `{"minimum": -13.922248386242977, "maximum": 0.2624383045243083, "mean": -6.5294308680372515, "median": -6.692502794410016}`
- RMS dBFS statistics: `{"minimum": -38.45485313217728, "maximum": -18.88753616157872, "mean": -25.667010952941254, "median": -25.801285938594553}`
- Silence-frame statistics: `{"minimum": 0.043859649122807015, "maximum": 0.7121212121212122, "mean": 0.34599174630983853, "median": 0.3482905982905983}`
- Leading-silence statistics: `{"minimum": 0.08, "maximum": 1.78, "mean": 0.308080229226361, "median": 0.3}`
- Trailing-silence statistics: `{"minimum": 0.0, "maximum": 2.84, "mean": 0.358305362259517, "median": 0.22}`
- SNR-proxy statistics: `{"minimum": 24.158103066378267, "maximum": 58.65084644769408, "mean": 41.8332084492048, "median": 41.91276033242738}`

These are conservative screening heuristics, not automatic deletion rules. Every
flagged sample remains in the raw dataset and should be reviewed by listening
before exclusion.
