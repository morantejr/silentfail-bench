# cand0 resume note (1 Oct 2026)

14 instances died in `SandboxTimeoutError` before any model call during the first generation
pass (DEFECTS #21). mini-swe-agent had written them into `preds.json` with an empty
`model_patch`, and its resume logic skips by `preds.json` key, so the resumed generator
skipped them. Those 14 entries were removed (backup:
`preds.json.pre_resume_with_14_empty_entries`) and the 14 instances were regenerated with
candidate 0's exact arguments (seed 1000, step_limit 100, same configs) in a parallel
process while the generator worked on candidates 1-3. Log: `gen_resume14.log`.
