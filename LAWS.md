# Laws

The properties ShaderIntel promises, each with the checker that verifies it. This file plays the
role Bend 2 gives `LAWS.bend` ("AGENTS.md backed by proof"): it is the specification, and every
line is executable — `SI.verify.proveAll(gpu, SI)` runs them all (the suite in `tests/` does so on
every run, see `tests/laws.test.js`), the way `bend PROOF.bend` checks a project before it ships.

The checkers live in `src/verify/index.js`. They are the small trusted base: a few lines of plain
WGSL or JavaScript each, sharing no code with the optimised kernels they judge (as Lean's kernel
re-checks what its elaborator and tactics produce, and Bend 2's `--verdict` re-checks proofs with
a kernel verified in Lean). A checker is either **exact**, **probabilistic** with a stated bound,
or a **physical law** with a tolerance — the column says which.

| Law | Statement | Checker | Kind |
|---|---|---|---|
| `scan` | `Scan.run` computes an exclusive prefix sum: s₀ = 0 and s[k+1] − s[k] = a[k] | `checkScan` | exact |
| `gemm` | `gemm` computes C = A·B | `freivalds` — A(Br) = Cr for random ±1 vectors r | probabilistic: false accept ≤ 2⁻⁸ with 8 trials |
| `gemm-rejects` | the certificate rejects a product with one corrupted entry | `freivalds` | exact (one entry ≥ 10⁻⁴ of its row's L2 norm is always caught) |
| `colouring` | no two dynamic bodies of one colour share a contact or a joint (what makes parallel Gauss–Seidel race-free) | `checkColoring` | exact |
| `contacts` | contact forces push (λₙ ≤ 0), friction stays in the isotropic Coulomb cone (‖λₜ‖ ≤ μ\|λₙ\|), and a settled 600-body pile of 20–60 g bodies penetrates less than 2 cm at 8 iterations | `checkContacts` | sign and cone exact; penetration with a tolerance (heavy-on-light stacks need substeps — see LEARNINGS) |
| `determinism` | a deterministic `RigidWorld` is a function of its inputs: two runs give bit-identical 64-bit state hashes | `checkDeterminism` | exact (same device, browser and driver) |
| `replay` | `restore(snapshot)` followed by the same steps reproduces the same state | `checkReplay` | exact |
| `mpm-conservation` | MLS-MPM particle-to-grid transfer conserves mass and momentum | `checkMPM` | exact up to the 2⁻²² fixed point |
| `backprop` | `MLP.backward` equals the central finite-difference gradient of the MSE loss | `checkGradient` | numerical (5% relative, f32) |

Further laws are checked by the test suite against analytic physics rather than by a checker
object: a pendulum's period (2π√(I/mgL), within 1%), a box on a 40° slope sliding at
g(sin θ − μ cos θ) (within 10%, measured 0.2%), a box sticking at 20°, balls and capsules rolling
10 s at constant speed (within 1%) and a free body spinning at 20 rad/s for 2 s, a large block resting
on 117 small boxes, Archimedes' waterline for floating boards (within ±0.15 of the submerged fraction
at a 3.1 cm grid), the smoke projection's divergence reduction, cloth strain, N-body energy.

## Contracts

Besides checking results, ShaderIntel checks its inputs where a type system would:

- **Binding contracts.** Every storage binding of a kernel knows the WGSL type it is declared with —
  structs included, parsed from the linked shader — and `bind()` rejects a buffer that is not a whole
  number of elements, naming the kernel, the binding, the byte size and the stride. A record written
  by JavaScript with the wrong number of fields (exactly the bug that once shifted every joint after
  the first by 16 bytes) is now an error at the call site instead of silent garbage on the GPU.
- **Uniform layouts** are generated from one declaration, so the bytes written from JavaScript and
  the struct read in WGSL cannot disagree.
- **Determinism as a property, not an accident.** The rigid solver is built so that no result depends
  on the order in which GPU threads run (sorted pair lists, prefix-sum allocation, double-buffered
  colouring, integer-only atomics): the confluence that interaction-net runtimes such as HVM (Bend 1)
  get from their semantics, obtained here by construction — and then checked by `determinism`.

## Transcripts (proof-carrying netcode)

A `RollbackSession` folds every confirmed tick's inputs into a hash chain and records the state hash
every 30 ticks. The transcript is a certificate of the match: `verifyTranscript` replays it from the
starting state and checks every state hash and the chain head. The test suite verifies an honest
transcript and rejects one with a single flipped input bit. When peers' hashes disagree, the session
repairs the divergence from the authority's state; the suite starts one peer from a nudged state and
checks that a single transfer brings every later hash back into agreement.
