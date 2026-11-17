# `pymdp.utils`

Retain the value returned by `validate_normalization`; the returned array
carries the validation dependency through compiled execution. Invalid eager
inputs raise `ValueError`. Traced validation uses Equinox's runtime error path,
so compiled failures need not have the same Python exception class.

::: pymdp.utils
