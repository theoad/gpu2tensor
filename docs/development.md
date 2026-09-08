# Development recipe

Build one complete example before generalizing an interface. Read the board,
state the acceptance check, implement the smallest useful change, run the check,
then record the result and the next limitation.

Independent work can be divided by ownership: CUDA execution/profiling, Trainium
execution/profiling, client/data APIs, and integration/evidence. The integration
owner controls shared request/result contracts and package files. Review another
backend's implementation against the same contract. Do not copy an earlier
experiment without checking its actual execution path and data ownership.

One Python source root is configured through pyproject.toml. Install editable in
a host-local virtual environment. IDEs should use that interpreter. All imports
are absolute. Vendor environments live on their respective workers; the Mac
client imports neither CUDA nor Neuron. Native code will get its own standalone
CMake project and compile_commands.json when needed; the first adapters call
existing vendor Python/native bindings.

Run `python -m pytest -q` with the learning, Gym and test extras installed.
Remove the generated `build/` tree before a release wheel build: setuptools can
otherwise retain files from a previous package-discovery configuration. Build
with `python -m pip wheel . --no-deps --wheel-dir dist`, inspect its entries,
and install it into a fresh environment outside the checkout. Tests and
`gpu2tensor.dev` operator tooling must be absent from wheels.

For a backend task, record the accepted source language, shape/dtype assumptions,
compile/run lifetime, correctness oracle, timing method, profiling overhead and
native artifact ownership. For a client task, check that observation-only use
imports no Gym/vendor runtime and consumes bounded memory. Integration owns the
real hardware example, negative control, provenance and worker cleanup. Pass
these facts and the exact SDK image to the next agent through the work board.
