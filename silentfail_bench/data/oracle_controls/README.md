# Oracle controls

Harness reports for the instrument checks described in the paper's instrument
section. They are small and kept because they are the evidence behind the
claim that the gold pipeline was validated before any model patch was graded.

- `NEG_CONTROL.negctl.json`, `NEG2.negctl2.json` — the no-op control: an empty
  patch submitted to the harness. `empty_patch_instances: 1`, `resolved: 0`.
  A gold pipeline that "resolves" an empty patch is broken.
- `gpt5mini_smoke.gpt5mini_smoke2.json` — end-to-end smoke: one real
  trajectory graded, `resolved: 1`.
- `gold.goldctl2/3/4.json` — earlier gold-control attempts that returned
  `error_instances: 1`. These are the artefacts of instrument failure #3 (the
  architecture-manifest pull failure), retained because the paper cites that
  failure.

Later oracle pairs — a reference patch that must resolve and an empty patch
that must not — are run inline by `modal/gold_modal.py` and recorded in each
run's `report.json` under `oracle`.
