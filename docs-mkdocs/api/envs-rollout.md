# `pymdp.envs.rollout`

At inference horizon one, learning uses a separate `learning_qs` carry and its
aligned transition action; truncating the inference window must not erase that
learning context. Warm starts derive missing learning context from supplied
`qs`, not a newly invented prior. Longer-window sufficient-statistic accounting
remains P2 and is not resolved by this horizon-one repair.

::: pymdp.envs.rollout
    options:
      members:
        - infer_and_plan
        - rollout
