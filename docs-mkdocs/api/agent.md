# `pymdp.agent`

`C` and `D` are batched independently of `A` and `B`. Use an explicit
rank-three representation for time-dependent `C`; ambiguous rank-two
time-versus-agent interpretation remains an owner decision (P9).

Missing-modality masks do not mutate the caller's arrays. Joint action support
and factor-marginal support are different contracts: retain the complete joint
support when indexing action probabilities. A-only learning does not require
transition actions; B learning still requires aligned actions.

::: pymdp.agent
    options:
      members:
        - Agent
