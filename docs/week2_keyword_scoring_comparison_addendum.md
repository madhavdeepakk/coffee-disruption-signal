## Root cause analysis (added after reviewing full results)

The initial interpretation above (written before I had the real results)
guessed the main failure mode would be false positives from keyword-only
overlap (festival pages, broker ads, etc.). That guess turned out to be
wrong, and the real result matters more:

- Agreement rate: 3/16 (19%) against Week 1's manual labels.
- False positives: 0. The score never rated something my manual labeling
  called Irrelevant as relevant-looking.
- False negatives: 10 of 16. The score badly under-rated articles I had
  manually labeled Relevant, scoring most of them exactly 0.000.

The cause is visible directly in the false-negative list: every one of
those 10 articles is in Russian, Korean, Portuguese, or Spanish (Bloomberg
Russian coverage of Brazil rain hitting the arabica market, a Korean
headline about a super El Niño hitting agricultural markets, multiple
Brazilian Portuguese headlines directly about coffee price spikes and
drought). Checked across the full 75-document sample, not just the 16
overlapping with Week 1 labels:

| Title script | Count | Avg keyword score | Scored exactly 0.000 |
|---|---|---|---|
| ASCII (mostly English) | 15 | 0.133 | 1/15 |
| Non-ASCII (other languages) | 60 | 0.016 | 48/60 (80%) |

The keyword scorer only checks for literal English words (`coffee`,
`drought`, `brazil`, plus the English-language `DISRUPTION_VOCAB` list). A
Portuguese article about "café" and "seca" (drought) and "Brasil" gets
essentially zero credit even when a bilingual reader can tell it is strong,
on-topic evidence. "Brasil" doesn't even match "brazil" (it's missing the
English "z" spelling), let alone "café" matching "coffee". This isn't a
scoring-threshold problem; it's a language-coverage gap in the method
itself.

Why this matters beyond Week 2: GDELT indexes global news in 65 languages,
and this sample shows the majority of directly relevant Brazil/coffee
coverage is in Portuguese (Brazil's own press, unsurprisingly) and other
non-English languages, not English. A relevance-scoring approach that
silently fails on non-English text would systematically under-retrieve the
sources most likely to have direct, local coverage of a Brazil-specific
disruption, which is the opposite of what this project needs. Since the zero
false positives here suggest the underlying idea (query terms + disruption
vocabulary) is directionally sound, the fix is coverage, not methodology.

Carried into Week 3/4/6, not fixed now (per the plan's scope boundaries:
FAISS/vector work is Week 3, gating logic is Week 6):

- Week 3's semantic retrieval should use a multilingual embedding model, not
  an English-only one. This finding is a concrete, data-backed reason why,
  not a hypothetical concern.
- A translation step (before or during scoring) is a plausible interim fix
  for keyword scoring specifically, worth prototyping in Week 3/4 rather
  than bolting onto Week 2's already-frozen baseline.
- Week 4's larger labeled sample should include non-English articles in
  proportion to what GDELT actually returns (this sample shows that's the
  majority, not a minority case) so the relevance threshold isn't tuned only
  against English text.
- This also means Week 1's small manual-labeling sample, done from
  translated headlines, was already doing "manual translation + relevance
  judgment", which is worth naming as a step the automated pipeline doesn't
  yet replicate.

On the earlier false-positive concern: it's not wrong that false positives
are the more dangerous failure mode long-term (an over-eager score feeding a
confident wrong explanation), and that's still why relevance gating exists.
But this run's main measured problem is under-retrieval of good evidence via
language coverage, not over-retrieval of bad evidence. Both matter; this
run's data only speaks to the second one.
