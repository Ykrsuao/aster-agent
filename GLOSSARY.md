# Aster

Aster agents combine deterministic Python capabilities with generation methods
whose instructions are carried out by an LLM.

## Language

**Execution cell**:
A unit of Python code executed during a generation method, with observations
and state available to subsequent cells. Its returned value is a cell observation;
the generation method's final result is a separate contract.
_Avoid_: Generation step, generation result

**Model route**:
A model connection identified by its model, destination, request interface,
and credential scope.
_Avoid_: Model name (when referring to a connection)

**Registry context**:
An alias's context-window declaration and the model route it describes.
It describes input capacity, separate from a response's requested output budget.
_Avoid_: Global model limit, request parameter

**Foreground turn**:
The work answering one accepted user prompt, including its steering input,
reply, and cancellation cleanup. A committed result seals steering; publishing
the final reply fixes the state restored with that reply. Background work may
continue after the turn finishes.
_Avoid_: Model call, background job
