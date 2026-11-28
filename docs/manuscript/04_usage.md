# Usage

In order to enhance the user-friendliness of `pymdp` without sacrificing flexibility, we have built the library to be highly modular and customizable, such that agents in `pymdp` can be specified at a variety of levels of abstraction with desired parameterisations. The methods of the `Agent` class can thus be called in any particular order, depending on the application, and furthermore they can be specified with various keyword arguments that entail choices of implementation details at lower levels.

By retaining a modular structure throughout the package's dependency hierarchy, `pymdp` also affords the ability to flexibly compose different low level functions. This allows users to customize and integrate their active inference loops with desired inference algorithms and policy selection routines. For instance, one could sub-class the `Agent` class and write a customized `step()` function, that combines whichever components of active inference one is interested in.
