import modal
from typing import Dict, Any

image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pulp", "fastapi[standard]")
    .run_commands("apt-get update && apt-get install -y glpk-utils")
)
app = modal.App("workforce-digital-twin-backend", image=image)

@app.function()
def run_simulation_task(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    from engine import LivingMonteCarloSimulator
    simulator = LivingMonteCarloSimulator(config=config_dict)
    return simulator.run_simulation()

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