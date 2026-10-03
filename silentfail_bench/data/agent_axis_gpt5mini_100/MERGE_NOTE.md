# predictions_merged.jsonl

114 rows. 97 from this directory's own run (6 workers); 17 regenerated at a
two workers (then one for any that rate-limited again) in `../agent_axis_gpt5mini_100_rerun_w1` because they exited `RateLimitError` on
OpenAI's TPM ceiling rather than on the model's own behaviour:
babel__babel-13928, babel__babel-14532, briannesbitt__carbon-2981, briannesbitt__carbon-3041, briannesbitt__carbon-3073, facebook__docusaurus-9183, fmtlib__fmt-2317, fmtlib__fmt-2457, gohugoio__hugo-12204, hashicorp__terraform-34900, immutable-js__immutable-js-2005, jqlang__jq-2681, laravel__framework-51890, micropython__micropython-12158, projectlombok__lombok-3009, prometheus__prometheus-9248, sharkdp__bat-2835.

Rerun exits: {'Submitted': 15, 'LimitsExceeded': 2}. Still empty after the rerun (counted as failures): ['babel__babel-14532', 'briannesbitt__carbon-2981'].
Counting a trajectory our rate limit killed as a capability failure would inflate the comparator's
failure rate; the Amendment 11 precedent (pilot_a11_gpt5mini/MERGE_NOTE.md) is followed exactly.
One gold run is made on the merged predictions as `a13_gpt5mini_100_merged`.
