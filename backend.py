import modal
from typing import Dict, Any
import json
import uuid
import os
import numpy as np

image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pulp", "fastapi[standard]")
    .run_commands("apt-get update && apt-get install -y glpk-utils")
    .add_local_python_source("engine")
)
app = modal.App("workforce-digital-twin-backend", image=image)

# Initialize Modal Volume for temporary file storage
results_volume = modal.Volume.from_name(
    "workforce-digital-twin-results",
    create_if_missing=True
)

@app.function(timeout=1800, cpu=2, memory=4096)
def run_trial_task(config_dict: Dict[str, Any], trial_number: int, trial_seed: int) -> Dict[str, Any]:
    from engine import LivingMonteCarloSimulator
    simulator = LivingMonteCarloSimulator(config=config_dict)
    return simulator.run_single_trial(trial_number, trial_seed)

@app.function(timeout=3600, volumes={"/results": results_volume})
def run_simulation_task(config_dict: Dict[str, Any], result_key: str) -> Dict[str, Any]:
    import time
    import pandas as pd
    from engine import LivingMonteCarloSimulator

    t_start = time.perf_counter()
    simulator = LivingMonteCarloSimulator(config=config_dict)
    trial_seeds = simulator.generate_trial_seeds()

    num_trials = simulator.cfg.trials
    configs = [config_dict] * num_trials
    trial_numbers = list(range(num_trials))

    trial_results = list(run_trial_task.map(configs, trial_numbers, trial_seeds))

    t_total = time.perf_counter() - t_start
    raw_results = simulator.aggregate_trial_results(trial_results, t_total)

    # --- PAYLOAD REDUCTION ---
    for key in ["employees", "projects", "performance_details", "simulation_summary"]:
        raw_results.pop(key, None)

    # Burnout Aggregation
    if raw_results.get("burnout"):
        df_burnout = pd.DataFrame(raw_results["burnout"])
        df_burnout = df_burnout.groupby(["trial", "step"])["rolling_fatigue"].mean().reset_index()
        df_burnout.rename(columns={"rolling_fatigue": "avg_rolling_fatigue"}, inplace=True)
        raw_results["burnout"] = df_burnout.replace({np.nan: None}).to_dict(orient="records")

    # Allocation Aggregation - Using strict schema from engine.py
    if raw_results.get("allocations"):
        df_alloc = pd.DataFrame(raw_results["allocations"])
        
        agg_dict = {
            "allocated_hours": "sum",
            "effective_output_hours": "mean",
            "role_match": "mean"
        }
        
        rename_dict = {
            "allocated_hours": "total_hours_allocated",
            "effective_output_hours": "avg_output_delivered",
            "role_match": "role_match_rate"
        }
        
        df_alloc = df_alloc.groupby(["trial", "step", "role"]).agg(agg_dict).reset_index()
        df_alloc.rename(columns=rename_dict, inplace=True)
        raw_results["allocations"] = df_alloc.replace({np.nan: None}).to_dict(orient="records")

    # --- SAVE TO MODAL VOLUME ---
    os.makedirs("/results", exist_ok=True)
    file_path = f"/results/{result_key}.json"

    with open(file_path, "w") as f:
        json.dump(raw_results, f)

    results_volume.commit()
    size_bytes = os.path.getsize(file_path)

    return {
        "result_key": result_key,
        "size_bytes": size_bytes
    }

@app.function()
@modal.fastapi_endpoint(method="POST")
def submit_simulation(config_dict: Dict[str, Any]):
    result_key = uuid.uuid4().hex
    call = run_simulation_task.spawn(config_dict, result_key)
    return {"status": "PENDING", "job_id": call.object_id}

@app.function()
@modal.fastapi_endpoint(method="GET")
def check_status(job_id: str):
    try:
        f_call = modal.FunctionCall.from_id(job_id)
        result = f_call.get(timeout=0)
        return {
            "status": "SUCCESS", 
            "result_key": result.get("result_key")
        }
    except TimeoutError:
        return {"status": "PENDING"}
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

@app.function(volumes={"/results": results_volume})
@modal.fastapi_endpoint(method="GET")
def get_result(result_key: str):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    
    results_volume.reload()
    
    file_path = f"/results/{result_key}.json"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/json")
    else:
        raise HTTPException(status_code=404, detail="Result not found")
