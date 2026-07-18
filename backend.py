import modal
from typing import Dict, Any

image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pulp", "fastapi[standard]")
    .run_commands("apt-get update && apt-get install -y glpk-utils")
    .add_local_python_source("engine")
)
app = modal.App("workforce-digital-twin-backend", image=image)

# One trial = one unit of parallel work. Modal will run many of these
# concurrently across separate containers.
@app.function(timeout=1800, cpu=2, memory=4096)
def run_trial_task(config_dict: Dict[str, Any], trial_number: int, trial_seed: int) -> Dict[str, Any]:
    from engine import LivingMonteCarloSimulator
    simulator = LivingMonteCarloSimulator(config=config_dict)
    return simulator.run_single_trial(trial_number, trial_seed)

# Orchestrator: fans trials out via .map(), then aggregates. Its own runtime
# is roughly "one trial's worth of time", not the sum of all trials, because
# the trials run in parallel underneath it.
@app.function(timeout=3600)
def run_simulation_task(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    import time
    from engine import LivingMonteCarloSimulator

    t_start = time.perf_counter()
    simulator = LivingMonteCarloSimulator(config=config_dict)
    trial_seeds = simulator.generate_trial_seeds()

    num_trials = simulator.cfg.trials
    configs = [config_dict] * num_trials
    trial_numbers = list(range(num_trials))

    trial_results = list(run_trial_task.map(configs, trial_numbers, trial_seeds))

    t_total = time.perf_counter() - t_start
    return simulator.aggregate_trial_results(trial_results, t_total)

@app.function()
@modal.fastapi_endpoint(method="POST")
def submit_simulation(config_dict: Dict[str, Any]):
    call = run_simulation_task.spawn(config_dict)
    return {"status": "PENDING", "job_id": call.object_id}

@app.function()
@modal.fastapi_endpoint(method="GET")
def check_status(job_id: str):
    try:
        f_call = modal.FunctionCall.from_id(job_id)
        result = f_call.get(timeout=0)
        return {"status": "SUCCESS", "result": result}
    except TimeoutError:
        return {"status": "PENDING"}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}
