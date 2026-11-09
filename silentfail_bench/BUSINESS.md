# SilentFail-Bench — Business Use Cases

SilentFail is a **selective trust / escalation layer** for critic-guided agents, not a standalone product.

## Where it plugs in

```text
candidates → selection critic (argmax) → commit trajectory
                                          ↓
                              disjoint audit panel → u
                                          ↓
                         M_c vs M_{c+u}  →  alert if high P(fail)
                                          ↓
                    human review / re-run / alternate critic / stop
```

## Use cases

| Setting | Business action |
|---|---|
| **Coding agents / PR bots** | High-$c$ patch selected by BoN; panel disagreement escalates before merge / CI-only trust |
| **Customer support / ops agents** | BoN-picked resolution; flag uncertain commits to supervisor queue under budget $b\%$ |
| **Enterprise agent platforms** | Cost control: review only top $b\%$ riskiest selected runs by $M_{c+u}$, not every trajectory |
| **Buying RMs / PRMs** | Vendor eval question: “After selection, does uncertainty add beyond score?” |
| **Safety / compliance** | Document that score-only gates miss failures; null result warns against over-trusting UQ |

## What “value” looks like (E7)

At fixed review budget $b$:

- Failures caught vs score-only  
- False alarms  
- Precision / recall @ $b$  
- Cost per additional failure caught  

Offline simulation first; adaptive recovery is out of scope for Stage 0–1.

## Rating context

Applied-AI fit is high when wired to escalation. Research value is the measurable protocol (SilentFail-Bench), not a new foundation model.
