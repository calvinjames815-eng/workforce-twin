# Enterprise Workforce Digital Twin

Monte Carlo simulation and mixed-integer optimization for large-scale workforce planning.

**Calvin James B. Demegillo**
[Live Demo](https://workforce-digital-twin.streamlit.app/) · [GitHub](https://github.com/calvinjames815-eng) · [LinkedIn](https://www.linkedin.com/in/calvin-james-demegillo-a1169a418/)

For an engineering design review of how this system evolved — architectural decisions, tradeoffs, and lessons learned — see [`CASE_STUDY.md`](./CASE_STUDY.md).

## Executive Summary

Enterprise Workforce Digital Twin is a simulation and optimization system that answers a question spreadsheets cannot: given uncertainty in demand, attrition, and fatigue, what is the *distribution* of workforce outcomes over the next several planning cycles, not just a single point estimate.

The system models a workforce of thousands of employees and hundreds of concurrent projects. At each planning cycle, a mixed-integer linear program (MILP) assigns employees to projects under availability, staffing, and fatigue constraints, while a workforce evolution model advances hiring, promotion, retirement, absence, and burnout. This full cycle is repeated across many independent Monte Carlo trials, producing a probabilistic picture of headcount adequacy, department-level bottlenecks, and burnout risk rather than a single deterministic forecast.

Three design decisions define the system:

1. **Optimization is embedded inside simulation, not run once.** Every planning cycle in every trial re-solves an assignment problem, so staffing decisions respond to the workforce state at that point in time rather than to a static snapshot.
2. **Execution is distributed, not local.** Monte Carlo trials are independent by construction, so they are dispatched to a serverless cloud backend (Modal) and executed in parallel instead of looping sequentially in a Streamlit process.
3. **Aggregation happens server-side.** Statistical summarization of trial results occurs in the backend before data reaches the client, keeping the dashboard responsive regardless of trial count.

The remainder of this document explains the architecture, the workforce and optimization models, the engineering tradeoffs behind these decisions, and how the system is deployed and operated.

## Problem Statement

Workforce planning at scale requires answering questions that are probabilistic and combinatorial in nature:

- Is current headcount sufficient to absorb the incoming project backlog?
- Which departments become bottlenecks first, and under what conditions?
- How much hiring is required to hold service levels, and by when?
- How does burnout propagate through the workforce as fatigue accumulates?
- How resilient is the plan to a sudden demand spike?

Spreadsheet-based planning answers these questions with static, single-scenario models that cannot represent stochastic attrition, project arrival variability, or the combinatorics of assignment under constraints. This project replaces that approach with a digital twin: a simulation model of the workforce coupled to an optimizer, run repeatedly under sampled uncertainty to produce a distribution of outcomes.

## Architecture

The system is split into three components, separated along a clear boundary: simulation and optimization logic, distributed execution, and presentation. This separation allows each layer to be scaled, tested, and replaced independently.

```
app.py (dashboard)
      |
      | async request
      v
modal.py (distributed execution layer)
      |
      | parallel trial dispatch
      v
engine.py (simulation + optimization core)
```

### engine.py — Simulation and Optimization Core

This module contains all domain logic and has no dependency on how it is invoked, which keeps it independently testable and reusable outside the web stack. Its responsibilities:

- Workforce and project generation
- Per-cycle workforce evolution (fatigue accumulation, promotion, retirement, hiring, absence)
- MILP construction and solving via PuLP/CBC for each planning cycle
- Monte Carlo trial orchestration
- Performance telemetry capture (solver time, cycle time, candidate counts)
- Result aggregation

Before constructing the MILP, the engine performs sparse candidate pruning: employee-project pairs that violate hard eligibility constraints (role mismatch, unavailability) are eliminated using vectorized NumPy operations before the optimization model is built. This keeps the constraint matrix sparse and the variable count bounded, which matters directly for solve time at the scale this system targets — thousands of employees and hundreds of projects per cycle, repeated across every trial and every cycle.

### modal.py — Distributed Execution Layer

This module exposes the simulation as an asynchronous service rather than a synchronous function call. Responsibilities:

- Accepts a simulation request and creates an asynchronous job
- Distributes Monte Carlo trials across parallel workers
- Aggregates trial results
- Compresses the result payload
- Persists results and exposes them through REST endpoints

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `submit_simulation` | Create a job and dispatch trials |
| GET | `check_status` | Poll job state |
| GET | `get_result` | Retrieve completed, aggregated results |

The backend is asynchronous end to end: submission returns immediately with a job identifier, and the frontend polls for completion. This means the dashboard never blocks a browser session on a long-running computation, and a simulation can outlive the client that requested it.

### app.py — Interactive Dashboard

A Streamlit application that lets users configure and inspect simulations without touching source code: Monte Carlo trial count, number of planning cycles, initial workforce size, and backlog size are all runtime parameters. It renders:

- KPI summaries
- Burnout and fatigue trends
- Department-level analytics
- Project-level analytics
- Solver and runtime performance telemetry

The dashboard is a pure consumer of the backend API. It holds no simulation state and performs no aggregation, which is a deliberate constraint explained below.

## Simulation Workflow

```
Configuration
      |
      v
Frontend submits simulation request
      |
      v
Modal creates an asynchronous job
      |
      v
Monte Carlo trials execute in parallel, each performing:
      - workforce evolution
      - MILP optimization
      - telemetry collection
      |
      v
Trial results aggregated server-side
      |
      v
Payload compressed and stored as JSON
      |
      v
Frontend polls job status
      |
      v
Frontend retrieves completed results
      |
      v
Dashboard renders analytics
```

## Workforce Model

Each simulated employee carries state that evolves over time:

- Role
- Availability
- Efficiency
- Experience and seniority
- Fatigue

Per-cycle transitions include promotion, retirement, absence, hiring, burnout onset, and efficiency degradation as a function of accumulated fatigue. These transitions are stochastic and independently sampled per trial, which is what allows the Monte Carlo layer to characterize a distribution of workforce trajectories rather than a single path.

## Optimization Model

At each planning cycle, the engine solves a MILP that assigns employees to projects.

- **Decision variables**: binary employee-to-project assignment for the current cycle.
- **Objective**: maximize total assignment utility, a function of role fit, efficiency, and fatigue state.
- **Constraints**:
  - Employee availability
  - Project staffing requirements
  - Maximum concurrent project assignments per employee
  - Cross-role assignment penalties
  - Fatigue-based penalties
  - Partial staffing penalties

Solving this exactly at full scale, for every cycle of every trial, is the dominant computational cost of the system. This is the reason candidate pruning happens before model construction rather than being left to the solver: reducing the variable and constraint count upstream is materially cheaper than relying on CBC's presolve to discover the same structure repeatedly across thousands of solves.

## Monte Carlo Engine

Each trial is fully independent:

- Independent RNG seed
- Independent workforce evolution trajectory
- Independent project arrival sequence
- Independent sequence of optimization solves

Independence is what makes parallel execution correct rather than merely convenient — trials share no mutable state, so distributing them across workers introduces no synchronization requirements. Results are aggregated statistically only after every trial completes.

## Engineering Decisions

**Parallel execution over sequential looping.** Monte Carlo trials are embarrassingly parallel by construction. Running them sequentially in a single Streamlit process wastes the available concurrency and ties wall-clock time directly to trial count. Dispatching trials to a distributed backend (Modal) trades operational simplicity for a linear-to-sublinear scaling relationship between trial count and runtime.

**Backend aggregation over frontend aggregation.** Aggregating thousands of trial results in the browser would require shipping raw per-trial data over the network and holding it in client memory, both of which scale poorly with trial count. Aggregating server-side means the client only ever receives summary statistics, decoupling dashboard responsiveness from simulation scale. The cost is that ad hoc, trial-level exploration in the frontend is not possible without a new backend endpoint — an accepted tradeoff given that the primary use case is distributional summaries, not individual trial inspection.

**Payload compression before serialization.** Reducing payload size before it crosses the network boundary lowers both transfer latency and storage cost, at the price of a small amount of CPU time in the backend. Given that compute is already distributed and elastic, this tradeoff favors compression.

**Fail-loud schema validation.** Simulation configuration and trial results are validated against an explicit schema, with invalid input or malformed intermediate state raising immediately rather than being silently coerced. In a system where a single malformed record could otherwise propagate through thousands of trials before surfacing in an aggregate statistic, failing at the point of ingestion is preferable to debugging a corrupted aggregate after the fact.

**Separate frontend and backend architecture.** Keeping `app.py` free of simulation and aggregation logic means the dashboard can be replaced, or additional clients can be built against the same API, without touching the simulation core. It also means `engine.py` can be tested, benchmarked, and versioned independently of the UI.

**Performance instrumentation as a first-class concern.** Solver time, per-cycle runtime, and candidate counts are recorded for every trial, not added ad hoc during debugging. At this scale, performance regressions are easy to introduce and hard to detect without a persistent, structured signal — the telemetry exists to make regressions visible in the dashboard rather than discovered in production runtimes.

## Performance Engineering

The system is bound by two costs that compound multiplicatively: the number of Monte Carlo trials and the number of planning cycles per trial, each requiring at least one MILP solve. Optimizations target both the per-solve cost and the surrounding orchestration overhead:

- Vectorized NumPy operations for candidate generation and eligibility filtering
- Candidate pruning and adjacency maps to keep the MILP sparse
- `argpartition` in place of full sorting where only top-k selection is needed
- Avoidance of unnecessary DataFrame copies in hot paths
- Dictionary-based aggregation in place of repeated DataFrame operations
- Parallel trial execution across the distributed backend
- Payload reduction prior to serialization
- Server-side aggregation in place of client-side aggregation
- Structured performance telemetry: solver timing and per-cycle runtime breakdown, surfaced in the dashboard

## Deployment

The system was originally implemented as a single-process Streamlit application, with simulation, optimization, and Monte Carlo orchestration all running synchronously in the same process as the UI. This was migrated to the current distributed architecture, moving simulation execution onto Modal as an asynchronous backend.

| | Single-process (original) | Distributed (current) |
|---|---|---|
| Trial execution | Sequential | Parallel across workers |
| Frontend memory | Holds full trial state | Holds only aggregated results |
| Scalability | Bound by one machine | Bound by backend concurrency |
| Execution model | Synchronous, blocking | Asynchronous, non-blocking |
| Compute | Local | Cloud (Modal) |

## Repository Structure

```
engine.py         Simulation and optimization core
modal.py          Distributed execution layer and REST API
app.py            Streamlit dashboard
requirements.txt  Dependencies
README.md
```

## Technology Stack

| Layer | Technology |
|---|---|
| Simulation / optimization | Python, NumPy, Pandas |
| MILP modeling and solving | PuLP, CBC |
| Distributed execution | Modal |
| API | FastAPI |
| Dashboard | Streamlit |

## Future Work

- GPU acceleration for candidate generation and vectorized workforce updates
- Distributed optimization for single-cycle MILPs that exceed practical serial solve time
- Scenario comparison across multiple simulation configurations
- Historical calibration of transition probabilities against real workforce data
- Sensitivity analysis on constraint parameters and objective weights
- Persisted, interactive scenario saving
- Explicit role hierarchy and skill-graph modeling for assignment eligibility
- Reinforcement-learning-based hiring policies as an alternative to fixed hiring rules

## Domain Coverage

This project spans operations research, mixed-integer optimization, stochastic simulation, distributed systems, cloud compute orchestration, backend API design, performance engineering, and applied data analytics, applied together to a single production-shaped system rather than in isolation.
