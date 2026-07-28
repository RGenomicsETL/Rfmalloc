# Execute a data-only architecture program

`rllm_execute()` interprets the dataflow and structured loops in an
[`rllm_program()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_input.md).
Semantic operators are ordinary R functions supplied in `operators`; a
small dense reference vocabulary is provided for arithmetic shared by
the architecture probes. This is the dense reference interpreter for the
same program that
[`rllm_forward()`](https://sounkou-bioinfo.github.io/Rfmalloc/Rllm/reference/rllm_forward.md)
binds to mapped weights and lowers natively through GGML.

## Usage

``` r
rllm_execute(
  program,
  inputs,
  parameters = list(),
  counts = list(),
  operators = list()
)
```

## Arguments

- program:

  A data-only `rllm_program`.

- inputs:

  A named list of input values. Extra entries remain available to
  operator functions through `context$inputs`.

- parameters:

  A named list containing a value for every declared program parameter.

- counts:

  A named list or atomic vector resolving symbolic loop counts and other
  symbolic integer attributes.

- operators:

  A named list of semantic operator functions.

## Value

A named list containing every program output and tap.

## Details

Each operator function is called with four named arguments: `inputs`,
the list of values arriving at the node; `attributes`, the node's
data-only attributes; `parameters`, the named parameter values; and
`context`, a list containing the current node and program, root and
local inputs, symbolic counts, and the enclosing loop iterations.
Operator functions control the representation of their results, so a
lowering may preserve a mapped or device-backed value rather than
materializing it in R.

The built-in dense reference vocabulary covers arithmetic,
normalization, pooling, activations and recurrence helpers. It also
covers ESM token dropout, key-padded rotary attention with attention-map
results, tied projection and contact regression, plus the dilated
convolution, inference batch normalization, cropping and softmax used by
OpenSpliceAI. Entries in `operators` replace built-ins with the same
name.
