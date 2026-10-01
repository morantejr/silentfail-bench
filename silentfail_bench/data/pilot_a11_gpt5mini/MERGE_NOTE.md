# predictions_merged.jsonl

41 rows. 38 from this directory's own run (6 workers); 3 regenerated at a single worker in
`../pilot_a11_gpt5mini_rerun3` because they exited `RateLimitError` on OpenAI's 500k TPM
ceiling rather than on the model's own behaviour: astropy__astropy-14539, django__django-10914, matplotlib__matplotlib-26208.

Counting a trajectory our rate limit killed as a capability failure would have inflated the
comparator's failure rate by up to 7.3 points and handed the Amendment 11 P11-2 gap to
DeepSeek on an instrument artefact. All three submitted on the rerun; one needed 86 steps.
