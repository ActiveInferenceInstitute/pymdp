# Sophisticated Inference Planning

Observations are integer category identifiers; categorical observations are
the default. Joint observation branches are bounded and normalized per
modality before pruning. Retained branches are not renormalized after pruning:
zero padding contributes no pruned-mass penalty, while real pruned branches do.
These contracts preserve the existing scientific convention rather than
introducing a new estimator, normalization policy, or search schedule.

The default SI adapter encodes imagined integer outcomes only when the agent has
no custom observation preprocessor. An agent's `preprocess_fn` receives integer
outcomes and owns their categorical conversion; the adapter does not encode them
twice. A custom inference callback likewise receives the original integer IDs.

::: pymdp.planning.si
