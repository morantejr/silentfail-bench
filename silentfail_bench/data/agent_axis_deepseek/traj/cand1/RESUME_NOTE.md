# cand1 resume note (1 Oct 2026, second pause)

66 instances died in `SandboxTimeoutError` before any model call: swerex waited for the
sandbox tunnel with Modal's 50 s default instead of the registered 600 s startup_timeout
(DEFECTS #22). mini-swe-agent had written them into `preds.json` with an empty `model_patch`;
those entries were removed (backup: `preds.json.pre_resume2_with_66_empty_entries`) so the
resumed generator regenerates them under this candidate's seed. Instances:
apache__druid-14092
apache__druid-15402
apache__druid-16875
apache__lucene-12196
apache__lucene-12212
apache__lucene-13494
astral-sh__ruff-15356
astral-sh__ruff-15443
astral-sh__ruff-15626
axios__axios-4731
axios__axios-5085
axios__axios-6539
babel__babel-13928
babel__babel-14532
babel__babel-15649
briannesbitt__carbon-2981
briannesbitt__carbon-3041
briannesbitt__carbon-3073
burntsushi__ripgrep-2209
burntsushi__ripgrep-2576
caddyserver__caddy-4943
caddyserver__caddy-5626
caddyserver__caddy-6051
facebook__docusaurus-8927
facebook__docusaurus-9183
facebook__docusaurus-9897
faker-ruby__faker-2705
faker-ruby__faker-2970
fastlane__fastlane-19304
fastlane__fastlane-20642
fastlane__fastlane-20958
fluent__fluentd-3616
fluent__fluentd-4030
fluent__fluentd-4311
fmtlib__fmt-2317
fmtlib__fmt-2457
fmtlib__fmt-3729
gin-gonic__gin-1805
gin-gonic__gin-3741
gin-gonic__gin-3820
gohugoio__hugo-12204
gohugoio__hugo-12343
gohugoio__hugo-12579
google__gson-1014
google__gson-2024
google__gson-2479
hashicorp__terraform-34814
hashicorp__terraform-34900
hashicorp__terraform-35611
immutable-js__immutable-js-2005
immutable-js__immutable-js-2006
javaparser__javaparser-4538
javaparser__javaparser-4561
jekyll__jekyll-8047
jekyll__jekyll-8761
jekyll__jekyll-9141
jordansissel__fpm-1829
jordansissel__fpm-1850
jqlang__jq-2681
jqlang__jq-2728
jqlang__jq-2919
laravel__framework-46234
laravel__framework-51890
laravel__framework-52451
micropython__micropython-10095
micropython__micropython-12158

## Resume 2026-10-01 22:35

38 placeholder entries (empty patch, no trajectory) removed so they regenerate:
axios__axios-4731
axios__axios-5085
axios__axios-6539
babel__babel-13928
babel__babel-14532
babel__babel-15649
briannesbitt__carbon-2981
briannesbitt__carbon-3041
briannesbitt__carbon-3073
burntsushi__ripgrep-2209
burntsushi__ripgrep-2576
caddyserver__caddy-4943
caddyserver__caddy-5626
caddyserver__caddy-6051
facebook__docusaurus-8927
facebook__docusaurus-9183
facebook__docusaurus-9897
faker-ruby__faker-2705
faker-ruby__faker-2970
fastlane__fastlane-19304
fastlane__fastlane-20642
fastlane__fastlane-20958
fluent__fluentd-3616
fluent__fluentd-4030
fluent__fluentd-4311
fmtlib__fmt-2317
fmtlib__fmt-2457
fmtlib__fmt-3729
gin-gonic__gin-1805
gin-gonic__gin-3741
gin-gonic__gin-3820
gohugoio__hugo-12204
gohugoio__hugo-12343
gohugoio__hugo-12579
google__gson-1014
google__gson-2024
google__gson-2479
hashicorp__terraform-34814

## Declared endpoint failures (2026-10-02 05:20 EDT)

hashicorp__terraform-34900 and valkey-io__valkey-790: two full attempts each ended in an uncapped OpenRouter retry loop with the sandbox idle (see ../ENDPOINT_FAILURES.json). Entered in preds.json with an empty patch and `endpoint_failure: true`; no trajectory exists. They are not regenerated.
