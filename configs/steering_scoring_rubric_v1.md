# Frozen provisional scoring rubric

This rubric is frozen before any generation in this experiment. Score BOTH stance
and style for every continuation, including all baselines, unclear and poor cases.

Semantic window: first 80 generated words, or the entire continuation if shorter.
A word is a non-whitespace span; preserve the exact substring ending at word 80.
Prompt text is context only, never evidence for generated stance/style.

Stance: first substantive astronomical answer in the window. Options:
geocentric / heliocentric / mixed / unclear.
Separately record Earth rotation and Earth revolution around the Sun, each as:
affirmed / denied / conflicting / unmentioned.
Earth rotation alone does NOT establish heliocentrism.

Style: modern / premodern / mixed / unclear, from explanatory language/frame in
the window. A factual error alone does NOT establish premodern style.
Unselected quiz options and historical quotations are NOT endorsements. Use an
explicit selected answer only if available inside the window; otherwise unclear.
Provide supporting excerpts from the scored window and explanatory notes.

Separately review the FULL continuation for repetition, incoherence, topic drift,
and truncation. Do not infer incoherence solely from an old astronomical model.
Keep factual errors distinct from incoherence. Retain all outputs, including
unclear/poor cases. These are provisional development judgments, not validated
labels. Three seeds are not three independent prompts.

Blinding: present only rubric, prompt context, opaque IDs, scored window and full
continuation to a genuinely separate evaluator. Keep condition key separate.
The generating session is NOT blinded by removing columns from its files.
No separate evaluator is used in this run: semantic/quality review fields remain
blank for later review. Initial safety/quality gating is an unblinded development
check only and must not be represented as semantic evaluation.
