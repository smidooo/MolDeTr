# Monitoring the scheduled lanes: an external dead-man's switch

`nightly.yml`, `integrations.yml` and `security.yml` all carry `schedule:` triggers, and all three
carry the same exposure: **GitHub disables a scheduled workflow after 60 days without a *commit* to
the repository** (issues, PRs, releases and tags do not count). A published paper's companion repo
goes quiet by design between releases, which is exactly the condition that trips this.

Nothing inside GitHub can notice this happening to itself. A workflow that watches another workflow
for staleness is itself a scheduled workflow, subject to the identical rule — it dies under the same
silence at the exact moment it would need to fire. That is why this is an *external* switch rather
than a fourth cron lane.

## Mechanism

Each of the three scheduled lanes ends (on success only) with a step using
`.github/actions/heartbeat-ping`, which pings a URL from a repository secret:

| Lane | Secret | Expected period |
|---|---|---|
| `nightly.yml` (`model` job) | `HEALTHCHECK_URL_NIGHTLY` | daily, `0 3 * * *` |
| `integrations.yml` (`external` job) | `HEALTHCHECK_URL_INTEGRATIONS` | weekly, `0 6 * * 1` |
| `security.yml` (`freshness-ping` job) | `HEALTHCHECK_URL_SECURITY` | weekly, `0 6 * * 3` |

The ping fires only when the lane's own guard steps also passed — see the comment on each
`heartbeat-ping` step for the exact success condition, since none of the three jobs is a flat
sequence of unconditional steps. A [healthchecks.io](https://healthchecks.io)-style service (or
equivalent) is configured with a check per lane, each set to the period above plus a grace window,
and alerts (email, or whatever the service supports) when a ping fails to arrive on schedule. That
alert is the thing that survives GitHub silently turning a lane off, a lane hanging past its
timeout, or a lane being deleted outright.

`.github/actions/heartbeat-ping` never fails the lane it reports on: an empty secret (e.g. a fork
with none configured) or a transient outage reaching the healthcheck service are both logged to the
step summary and swallowed, never propagated as a job failure. Letting a monitoring hiccup redden a
real green run would train people to ignore red runs, which is the opposite of the point.

## Setting it up (one-time, human)

1. Create an account at a dead-man's-switch service (healthchecks.io's free tier covers this: three
   checks, generous grace periods). This is an external account this project does not manage —
   analogous to the Hugging Face and Zenodo accounts in `deploy/EXTERNAL_STEPS.md`.
2. Create three checks, one per lane above, each with a period matching its cron schedule and a
   grace window wide enough to absorb normal run duration plus a missed-then-retried dispatch
   (a few hours is generous for the daily lane, a day or two for the weekly ones).
3. Add the three ping URLs as repository secrets: `HEALTHCHECK_URL_NIGHTLY`,
   `HEALTHCHECK_URL_INTEGRATIONS`, `HEALTHCHECK_URL_SECURITY` (Settings → Secrets and variables →
   Actions).
4. Verify the switch actually fires before trusting it: pause one check in the service's UI, wait
   past its grace period, and confirm the alert arrives. A monitor nobody has seen fire is not a
   monitor — the same discipline this repo already applies to every other guard.

Until the secrets are set, every ping step visibly reports "no healthcheck configured" in its job's
step summary rather than silently doing nothing — check there if a lane's freshness state is in
doubt.

## What this does not cover

- It reports that a scheduled run **happened and its own guard steps passed**, not that the checks
  those steps run are themselves sufficient — `tests/test_integrations_isolation.py` and the
  "fail if the lane did not actually run the tests" step in `nightly.yml` are what keep the guards
  honest; this switch only says the lane still executes at all.
- It does not distinguish "GitHub disabled the schedule" from "the workflow file has a syntax error"
  from "the runner queue is starved" — any of these stops the ping, and the alert is the same. Given
  how rarely this fires, that ambiguity is acceptable; the first move on an alert is to check the
  Actions tab for the lane in question.
- A third scheduled workflow added later without a `heartbeat-ping` step is caught by
  `tests/test_workflow_freshness.py`, not by this document — the test is the enforcement, this file
  is the explanation.
