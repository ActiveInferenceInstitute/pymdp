# `pymdp.inference`

Single-frame VMP accepts absent or empty action history and retains the
provided prior. This state-inference contract does not decide longer-window
learning sufficient-statistic accounting (P2).

::: pymdp.inference
    options:
      members:
        - update_posterior_states
        - joint_dist_factor
        - smoothing_ovf
        - smoothing_exact
