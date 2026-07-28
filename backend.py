import modal
import json
import uuid
import os
import secrets
import logging
import time
from typing import Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from fastapi import HTTPException, Security, Depends, Query
from fastapi.security import APIKeyHeader
from fastapi.responses import FileResponse
import pandas as pd

logger = logging.getLogger(__name__)

# --- API KEY AUTHENTICATION ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    expected_key = (
        os.environ.get("X_API_KEY")
        or os.environ.get("INTERNAL_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not expected_key:
        logger.error("API Key secret is not configured in backend environment variables.")
        raise HTTPException(status_code=500, detail="Server security configuration error.")

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API Key")
    return api_key

# --- RATE LIMITING (PORTFOLIO DEPLOYMENT, BEST-EFFORT) ---
rate_limit_dict = modal.Dict.from_name("workforce-rate-limits", create_if_missing=True)
MAX_JOBS_PER_MINUTE = 5

def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    now = time.time()
    timestamps = rate_limit_dict.get(api_key, [])
    valid_timestamps = [ts for ts in timestamps if now - ts < 60]

    if len(valid_timestamps) >= MAX_JOBS_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {MAX_JOBS_PER_MINUTE} jobs per minute allowed."
        )

    valid_timestamps.append(now)
    rate_limit_dict[api_key] = valid_timestamps
    return api_key

# --- REQUEST SCHEMAS ---
class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trials: int = Field(50, ge=5, le=200, description="Monte Carlo Trials")
    steps_per_trial: int = Field(12, ge=4, le=24, description="Planning Cycles Per Trial")
    initial_employees: int = Field(100, ge=20, le=5000, description="Initial Workforce Size")
    initial_projects: int = Field(40, ge=10, le=1000, description="Backlog Pipeline Size")

# --- MODAL IMAGE & APP ---
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pulp", "fastapi[standard]", "pydantic")
    .add_local_python_source("engine")
)
app = modal.App("workforce-digital-twin-backend", image=image)

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
    from engine import LivingMonteCarloSimulator
    import time

    t_start = time.perf_counter()
    simulator = LivingMonteCarloSimulator(config=config_dict)
    trial_seeds = simulator.generate_trial_seeds()

    num_trials = simulator.cfg.trials
    configs = [config_dict] * num_trials
    trial_numbers = list(range(num_trials))

    trial_results = list(run_trial_task.map(configs, trial_numbers, trial_seeds))

    t_total = time.perf_counter() - t_start
    raw_results = simulator.aggregate_trial_results(trial_results, t_total)

   
    raw_allocations = raw_results.get("allocations", [])
    allocation_summary_records = []
    if raw_allocations:
        df_alloc = pd.DataFrame(raw_allocations)
        group_cols = [c for c in ["trial", "step", "role"] if c in df_alloc.columns]
        if group_cols and not df_alloc.empty:
            df_alloc_summary = df_alloc.groupby(group_cols).agg(
                allocated_count=("emp_id", "count"),
                total_hours_allocated=("allocated_hours", "sum"),
                avg_output_delivered=("effective_output_hours", "mean"),
                role_match_rate=("role_match", "mean"),
            ).reset_index()
            allocation_summary_records = df_alloc_summary.to_dict(orient="records")

    
    raw_results["allocation_summary"] = allocation_summary_records

  
    keys_to_prune = ["employees", "projects", "performance_details", "simulation_summary", "allocations"]
    for key in keys_to_prune:
        raw_results.pop(key, None)

    
    if "burnout" in raw_results and raw_results["burnout"]:
        df_burnout = pd.DataFrame(raw_results["burnout"])
        if not df_burnout.empty:
            df_burnout_agg = df_burnout.groupby("step")["rolling_fatigue"].mean().reset_index()
            df_burnout_agg.rename(columns={"rolling_fatigue": "avg_rolling_fatigue"}, inplace=True)
            df_burnout_agg["avg_rolling_fatigue"] = df_burnout_agg["avg_rolling_fatigue"].apply(lambda x: round(float(x), 4))
            raw_results["burnout"] = df_burnout_agg.to_dict(orient="records")

    os.makedirs("/results", exist_ok=True)
    file_path = f"/results/{result_key}.json"

    with open(file_path, "w") as f:
        json.dump(raw_results, f)

    results_volume.commit()
    size_bytes = os.path.getsize(file_path)

    return {"result_key": result_key, "size_bytes": size_bytes}

# --- REST ENDPOINTS ---

@app.function(secrets=[modal.Secret.from_name("digital-twin-secrets")])
@modal.fastapi_endpoint(method="POST")
def submit_simulation(payload: SimulationRequest, api_key: str = Depends(check_rate_limit)):
    result_key = uuid.uuid4().hex
    config_dict = payload.model_dump()
    call = run_simulation_task.spawn(config_dict, result_key)
    return {"status": "PENDING", "job_id": call.object_id}

@app.function(secrets=[modal.Secret.from_name("digital-twin-secrets")])
@modal.fastapi_endpoint(method="GET")
def check_status(job_id: str, api_key: str = Depends(verify_api_key)):
    try:
        f_call = modal.FunctionCall.from_id(job_id)
        result = f_call.get(timeout=0)
        return {"status": "SUCCESS", "result_key": result.get("result_key")}
    except TimeoutError:
        return {"status": "PENDING"}
    except Exception as e:
        logger.error(f"Error checking status for job '{job_id}': {str(e)}", exc_info=True)
        return {"status": "FAILED", "error": "Simulation execution failed on worker process."}

@app.function(volumes={"/results": results_volume}, secrets=[modal.Secret.from_name("digital-twin-secrets")])
@modal.fastapi_endpoint(method="GET")
def get_result(
    result_key: str = Query(..., pattern="^[a-f0-9]{32}$", description="Strict 32-character hex UUID"),
    api_key: str = Depends(verify_api_key)
):
    results_volume.reload()
    file_path = f"/results/{result_key}.json"
    normalized_path = os.path.normpath(file_path)

    if not normalized_path.startswith("/results/"):
        raise HTTPException(status_code=400, detail="Invalid result identifier path.")

    if os.path.exists(normalized_path):
        return FileResponse(normalized_path, media_type="application/json")
    else:
        raise HTTPException(status_code=404, detail="Result payload not found or expired.")

@app.function(volumes={"/results": results_volume}, schedule=modal.Cron("0 0 * * *"))
def cleanup_old_results():
    results_volume.reload()
    now = time.time()
    if os.path.exists("/results"):
        for filename in os.listdir("/results"):
            file_path = os.path.join("/results", filename)
            if os.path.isfile(file_path) and (now - os.path.getmtime(file_path) > 86400):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {str(e)}")
        results_volume.commit()
