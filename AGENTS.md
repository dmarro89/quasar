# Quasar Development Guidelines

Quasar is built incrementally to teach how an inference engine works from first principles. Every change should keep the code small, measurable, and easy to understand.

## Toolchain

- Use the latest stable Go release available at development time.
- Keep `go.mod` and CI aligned with the current stable Go toolchain.
- Prefer the Go standard library. Add an external dependency only when it removes substantial complexity or provides functionality we should not reasonably maintain ourselves; explain the reason in the PR.

## Design

- Prefer small, cohesive packages with explicit responsibilities.
- Use structs to model state. Introduce interfaces only when there are multiple implementations or a concrete testing/design need.
- Avoid global mutable state, hidden side effects, and unnecessary abstraction.
- Keep APIs and data flow obvious. Educational clarity is a project requirement.
- Return explicit errors rather than silently recovering from invalid state.

## Performance

- Consider CPU cost, memory usage, allocations, copies, and data layout in every implementation.
- Correctness and clarity come before micro-optimization.
- Establish a measurable baseline before optimizing hot paths.
- Add benchmarks when a code path becomes performance-sensitive, and keep optimizations only when measurements justify them.

## Testing

- Every behavior change must be unit tested.
- Tests must be deterministic and independent of network access.
- Before merging, run formatting, `go vet ./...`, and `go test ./...`.
- Add benchmarks alongside performance-sensitive components when they are introduced.

## Pull Requests

- One concept or behavior change per PR.
- Keep PRs small and avoid unrelated refactors.
- Prefer a small number of changed files.
- Update the README when user-visible behavior or the current release changes.
- Do not mix cleanup with functional changes unless the cleanup is required by that change.

## Documentation and Releases

- Use semantic versioning.
- Every release gets `docs/releases/vX.Y/README.md` describing the goal, concepts introduced, implementation, tests/benchmarks, limitations, and the connection to a real inference engine.
- General engine documentation lives separately under `docs/` and evolves toward the final architecture documentation.
- The root `README.md` must always describe the latest released version.
