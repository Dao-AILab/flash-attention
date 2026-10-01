# Quack persistent-cache fingerprint test result

Baseline: `616b0e8abab13b87b01525b3916d5a863ab02ae0`.

| Host source-function check | Baseline | Patch |
| --- | --- | --- |
| Unchanged Quack version retains the fingerprint | Passed | Passed |
| Changing only the Quack version invalidates the fingerprint | Failed | Passed |
| Cutlass version change invalidates the fingerprint | Preserved | Passed |
| TVM-FFI version change invalidates the fingerprint | Preserved | Passed |
| Missing Quack distribution metadata | Ignored | `PackageNotFoundError` |

These checks execute the production fingerprint function body with fixed
runtime-version stubs. They do not import the complete FA4 stack or execute
GPU kernels. The patch keeps process-local memoization and propagates the
standard missing-metadata error. Disabled persistent caching does not request
the fingerprint.

## Speed comparison

| Fresh-process workload | First run | Reuse run | Relative speed | Status |
| --- | ---: | ---: | ---: | --- |
| H20 forward/backward, Quack stamp A | — | — | — | GPU execution pending |
| H20 forward/backward, changed Quack stamp B | — | — | — | GPU execution pending |

No steady-state inference speedup is expected from adding a dependency stamp.
The GPU lifecycle check must demonstrate cache export, disk reuse, invalidation
and reuse after invalidation in separate processes, on the same patched source.
Changing a test metadata stamp represents simulated version identity, not a
real Quack upgrade. Existing cache tests and six new pytest cases remain
unexecuted because the provisioned environment lacks FA4's required packages.

The contract covers installed distribution versions between fresh processes.
Same-version source edits and dependency replacement inside a running process
are outside this change.
