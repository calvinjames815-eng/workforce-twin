# Engineering Case Study: Enterprise Workforce Digital Twin

A design review of how a workforce planning simulator evolved from a single-process prototype into a distributed, asynchronous simulation system.

**Calvin James B. Demegillo**
[Live Demo](https://workforce-digital-twin.streamlit.app/) · [GitHub](https://github.com/calvinjames815-eng/workforce-twin) · [LinkedIn](https://www.linkedin.com/in/calvin-james-demegillo-a1169a418/)

## Executive Summary

Workforce planning is a decision problem under uncertainty, and most organizations solve it with tools built for certainty. This project began as an attempt to close that gap: combine stochastic simulation with mixed-integer optimization so that workforce decisions could be evaluated against a distribution of futures rather than a single spreadsheet scenario.

The initial implementation was a single-process Streamlit application that ran simulation, optimization, and aggregation sequentially in one Python process. It worked, and it also failed in a predictable way: runtime scaled linearly with the number of Monte Carlo trials, the UI blocked for the full duration of every run, and the design had no path to handle the trial counts a real planning exercise would require.

The system was redesigned around three principles: separate simulation logic from execution orchestration, distribute independent work instead of looping over it, and move aggregation to the backend so the frontend's cost is bounded by result size, not trial count. The result is an architecture where Monte Carlo trials execute in parallel on a distributed backend, the dashboard polls asynchronously instead of blocking, and payloads are aggregated and compressed before they ever reach the browser.

This document is a design review of that evolution — the problems that forced it, the tradeoffs accepted along the way, and what the architecture can and cannot do today.

## 1. Background

### Why workforce planning is hard

Workforce planning sits at the intersection of two kinds of difficulty. The first is combinatorial: assigning the right people to the right projects under availability, skill, and capacity constraints is an assignment problem, and assignment problems at the scale of a real organization — thousands of employees, hundreds of concurrent projects — are not solvable by inspection or by hand-built spreadsheet logic. The second is stochastic: attrition, hiring timelines, demand arrival, and fatigue accumulation are not known quantities. They are distributions.

### Why spreadsheets fail

Spreadsheet-based planning treats both of these difficulties as if they were absent. It typically encodes a single scenario — one attrition rate, one demand forecast, one staffing plan — and presents its output as if it were a forecast rather than one sample from a much wider space of possible outcomes. This works acceptably when the underlying system is stable and slow-moving. It fails precisely when planning matters most: under demand volatility, elevated attrition, or capacity stress, which is exactly when decision-makers need to see a range of outcomes and their relative likelihood, not a single number.

### Why deterministic optimization alone is insufficient

Replacing a spreadsheet with a single optimization run — solve once, staff optimally, done — fixes the combinatorial problem but not the stochastic one. A single MILP solve produces the optimal assignment for one realization of the future. It says nothing about how sensitive that plan is to a slightly worse attrition quarter, or what the staffing gap looks like in the tail scenarios that actually cause organizational pain. Optimization answers "what is the best plan for this scenario." It does not answer "how many scenarios does this plan fail in."

### Why Monte Carlo simulation was introduced

Monte Carlo simulation was introduced to answer that second question. Instead of solving the assignment problem once, the system solves it once per planning cycle, inside each of many independently sampled trials — each trial with its own attrition draws, hiring outcomes, and fatigue trajectories. The result is not a single staffing plan but a distribution of outcomes: a picture of how often a given headcount is sufficient, where bottlenecks concentrate, and how burnout risk evolves across plausible futures rather than one assumed one.

## 2. Initial Architecture

The first version of the system was intentionally simple: a single `engine.py` module containing simulation and optimization logic, invoked directly by a single Streamlit application. There was no backend service, no job queue, and no separation between computation and presentation.

```
Streamlit process
  |
  |-- generate workforce
  |-- for each Monte Carlo trial (sequential):
  |       for each planning cycle:
  |           evolve workforce
  |           solve MILP
  |           record telemetry
  |-- aggregate results in-process
  |-- render dashboard
```

This architecture had one property that made it a reasonable starting point and one property that made it unsustainable. The reasonable property: everything executed in a single process with no network boundary, so there was nothing to debug except the simulation logic itself. The unsustainable property: every Monte Carlo trial ran on the same thread, one after another, and the user's browser tab was blocked for the entire duration.

Runtime scaled linearly with trial count. A configuration that felt responsive at ten trials became a multi-minute blocking operation at a few hundred, and a planning exercise that would benefit from thousands of trials — the regime where Monte Carlo estimates actually stabilize — was simply not usable interactively. The architecture did not have a scaling path; it had a ceiling.

## 3. Engineering Challenges

Several distinct problems surfaced as trial counts and workforce size increased, and they compounded rather than occurring in isolation.

| Challenge | Cause | Consequence |
|---|---|---|
| MILP cost per solve | Full employee-project candidate space passed to the solver unfiltered | Solve time grew faster than workforce size |
| Runtime multiplication | Every trial re-solves a MILP at every planning cycle | Total solves = trials × cycles, compounding any per-solve inefficiency |
| Candidate explosion | No upstream filtering of infeasible employee-project pairs | Constraint matrix grew denser than necessary, slowing presolve and solve |
| Large payloads | Raw per-trial results held for later aggregation | Memory pressure and slow serialization at higher trial counts |
| Frontend memory growth | Aggregation performed in the Streamlit process | Browser tab process footprint grew with trial count, not with result size |
| Network transfer cost | Uncompressed results moved between components | Higher latency and higher storage cost as result volume grew |
| Solver scalability | CBC solve time is sensitive to model size and structure | Sequential solves left no way to amortize this cost across trials |
| UI responsiveness | Synchronous execution model | The interface had no way to remain interactive during a run of any meaningful size |

None of these were solved by a single fix. They required rethinking where computation happened, what data moved between components, and when.

## 4. Architectural Evolution

The redesign followed a specific sequence, and each step addressed a specific failure mode from the prior architecture rather than being a general-purpose rewrite.

```
Single-process Streamlit
        |
        v
Modal backend introduced  ---------------  decouples execution from the UI process
        |
        v
Parallel trial execution  ---------------  removes the linear runtime-vs-trials relationship
        |
        v
Backend aggregation       ---------------  bounds frontend cost by result size, not trial count
        |
        v
Asynchronous polling       --------------  removes UI blocking during execution
        |
        v
Compressed payloads         -------------  reduces transfer latency and storage cost
```

**Modal backend.** Moving execution off the Streamlit process was the precondition for everything that followed. As long as simulation ran inside the same process rendering the UI, no amount of internal optimization would fix the blocking problem — the browser tab and the computation were coupled by construction. Introducing a separate execution layer decoupled them, at the cost of introducing a network boundary and the operational complexity that comes with it: job state, polling, and failure handling that did not exist before.

**Parallel trial execution.** Because Monte Carlo trials are independent by construction — no trial reads or writes another trial's state — they were a natural fit for parallel dispatch once execution moved to a distributed backend. This is the change that broke the linear runtime-vs-trial-count relationship. The tradeoff is that parallel execution introduces variance in job completion time and requires the backend to track many concurrent units of work instead of one sequential loop.

**Backend aggregation.** Once trials ran in parallel, someone still had to combine their results into summary statistics. Doing this in the frontend would mean shipping every trial's raw output across the network and holding all of it in browser memory — the same growth problem as before, just relocated. Aggregating server-side means the frontend only ever receives a fixed-size summary regardless of trial count. The tradeoff is that any analysis not covered by the existing aggregation logic requires a backend change; the frontend cannot arbitrarily reprocess raw trial data it never received.

**Asynchronous polling.** With execution now happening remotely and in parallel, the frontend no longer needed to block at all — it could submit a job, receive an identifier, and poll for completion. This is what actually restored interactivity: the browser tab remains responsive for the full duration of a run, including runs long enough that blocking would have made the UI unusable regardless of how fast the backend was.

**Compressed payloads.** The final step addressed transfer and storage cost directly: results are compressed before they leave the backend. This trades a small, fixed amount of backend CPU time for a reduction in network transfer time and storage footprint — a favorable trade given that compute in this architecture is already elastic and distributed, while network and storage costs scale with every request.

## 5. Optimization Decisions

Distributing execution addressed the *orchestration* cost of running many trials in parallel. It did not address the *per-solve* cost of the MILP itself, which is where a separate set of optimizations was applied.

**Vectorization.** Candidate generation and eligibility filtering were rewritten using NumPy array operations instead of row-wise iteration. This matters because eligibility filtering runs once per cycle per trial — any per-row Python overhead is multiplied by cycle count and trial count, making it one of the highest-leverage places to remove interpreter overhead.

**Candidate pruning.** Employee-project pairs that violate hard eligibility constraints are removed before the MILP is constructed, rather than being left in the model for the solver to discover as infeasible. This keeps the constraint matrix sparse from the start, which reduces both model construction time and solver time — CBC does not need to rediscover structure that can be established cheaply upstream.

**Adjacency maps.** Precomputed adjacency structures between employees and eligible projects avoid recomputing eligibility relationships repeatedly within a cycle, trading a small amount of memory for reduced repeated computation.

**`argpartition` instead of full sorting.** Several selection steps only need the top-k candidates, not a fully ordered list. Using `argpartition` instead of a full sort reduces the algorithmic cost of these steps from O(n log n) to O(n), which is a meaningful difference at the candidate volumes this system processes per cycle.

**Dictionary-based aggregation.** Certain aggregation steps use plain dictionaries instead of repeated DataFrame operations, avoiding the overhead DataFrames carry for small, high-frequency updates — overhead that is negligible once but not negligible when repeated across every cycle of every trial.

**Avoiding DataFrame copies.** Pandas operations that implicitly copy data were identified and replaced with in-place or view-based alternatives where correctness allowed it, reducing both memory pressure and the CPU cost of repeated copying in hot paths.

**Performance telemetry.** Solver time, cycle time, and candidate counts are recorded for every trial rather than added only when a performance problem is suspected. The reasoning: in a system where a regression can only be observed indirectly — as a slower dashboard, not a explicit error — instrumentation is what makes the regression visible at all, and it needs to already exist before the regression occurs, not after.

## 6. Results

The architectural changes produced qualitative improvements consistent with their design intent. No formal load-testing benchmarks were conducted, so the results below are described qualitatively rather than with fabricated figures.

- **Frontend memory usage** is materially lower because the browser process now holds aggregated summaries rather than raw per-trial results.
- **Monte Carlo trials execute in parallel**, removing the direct linear relationship between trial count and wall-clock runtime that existed in the original design.
- **The dashboard remains responsive during execution**, since the frontend no longer blocks on simulation and instead polls a job status endpoint.
- **The architecture is distributed rather than single-process**, which gives it a scaling path — additional backend capacity can absorb larger trial counts — that the original design structurally lacked.
- **Concerns are cleanly separated**: simulation and optimization logic in `engine.py`, orchestration and API surface in `modal.py`, and presentation in `app.py`. Each can be modified, tested, or replaced with limited impact on the others.

## 7. Lessons Learned

**Simulation architecture.** Keeping simulation logic free of any dependency on how it is invoked — no references to Streamlit, no references to the API layer — made it possible to move that logic to a distributed backend without rewriting it. Domain logic and execution context should be separable from the start, not separated retroactively.

**Distributed systems.** Parallelism is only free when units of work are truly independent. The Monte Carlo trial structure made this straightforward here, but it is a property of the problem, not a property of distributed systems in general — most workloads require explicit design work to reach that independence.

**Optimization.** The most effective MILP performance work happened before the solver was invoked, not inside it. Reducing candidate count and constraint density upstream had more impact than any solver-level tuning would have.

**API design.** An asynchronous submit/poll/retrieve pattern is a small amount of additional surface area that removes an entire class of UI problems — there was no version of a responsive interface that kept a synchronous execution model.

**Data contracts.** Once execution and presentation are separated by a network boundary, the shape of the data crossing that boundary becomes a first-class design decision, not an implementation detail. Aggregation decisions made in the backend directly determine what the frontend is capable of showing.

**Backend/frontend separation.** This paid off exactly at the moment the frontend needed to stop doing work it was previously doing — aggregation, primarily. A boundary that did not previously matter for correctness turned out to matter a great deal for scalability.

**Performance engineering.** Optimizations aimed at a system that runs the same computation thousands of times (once per cycle, per trial) look different from optimizations aimed at a system that runs once. Overhead that is irrelevant in isolation is not irrelevant when multiplied by cycle count and trial count.

**Observability.** Telemetry was worth building before it was needed. In a system with this much multiplicative cost, the alternative — diagnosing a slowdown with no historical signal — is materially harder.

**Schema validation.** Validating configuration and intermediate results at ingestion, rather than allowing malformed data to propagate, meant failures surfaced at their source instead of as a corrupted aggregate several stages downstream.

## 8. Future Work

The architecture described here is enterprise-*style*, not enterprise-*scale* — it has not been deployed against production organizational data or operated under sustained real-world load. The following areas extend it in that direction:

- **Distributed optimization.** For MILP instances that exceed practical single-solver solve time, distributing the optimization itself — not just the trials — across workers (e.g., Benders decomposition or column generation) would remove the current constraint that each cycle's MILP is solved on a single node.
- **Scenario management.** Persisting and comparing named scenarios, rather than treating each simulation as ephemeral, would support the comparative analysis planning teams actually need.
- **Historical calibration.** Fitting transition probabilities (attrition, promotion, absence) to real historical workforce data rather than assumed distributions would move the model from illustrative to predictive.
- **Real enterprise data integration.** Connecting to actual HRIS or ERP data sources, with the schema validation and data contract discipline already established, rather than synthetic workforce generation.
- **Container autoscaling and Kubernetes.** The current backend relies on Modal's managed scaling; a Kubernetes-based deployment would allow more explicit control over resource allocation and cost at higher sustained load.
- **Cloud deployment hardening.** Authentication, multi-tenant isolation, and cost controls appropriate for a system handling real organizational data, none of which are currently implemented.
- **GPU acceleration.** Candidate generation and workforce vector updates are numerically dense operations that are plausible GPU offload candidates at larger workforce sizes.
- **Role hierarchies and skill graphs.** Replacing flat role matching with a structured hierarchy or graph would make eligibility and substitution modeling considerably more realistic.
- **Reinforcement-learning hiring policies.** Replacing fixed hiring rules with a learned policy is a natural extension once enough calibrated historical data exists to train against.

## 9. Reflection

The most useful part of building this system was not the optimization model or the simulation logic individually — it was the experience of watching a working, single-process design become the wrong design as scale requirements increased, and having to decide what to change and what to leave alone.

Every architectural change here was a tradeoff, not a strict improvement: distribution added operational complexity in exchange for parallelism; backend aggregation reduced frontend flexibility in exchange for bounded memory growth; asynchronous polling added state management in exchange for a responsive UI. None of these were free, and part of the engineering work was recognizing which costs were acceptable given what the system needed to do.

Building this pushed the work past algorithms and into systems: correctness of the MILP formulation mattered, but so did what happened when that formulation had to run thousands of times under a UI that could not afford to block. That combination — getting the model right and getting the system around it right — is the part of the project most directly transferable to production engineering work, and it is the part this case study has tried to make explicit rather than leave implicit in the code.
