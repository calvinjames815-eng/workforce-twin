import modal
import App
from typing import Dict, Any

# 1. Define the isolated runtime environment for the worker container
# Modal automatically builds this image and deploys it to the cloud
image = (
    modal.Image.debian_slim()
    .pip_install("pandas", "numpy", "pulp", "fastapi[standard]")
    .run_commands("apt-get update && apt-get install -y glpk-utils")
)

# Instantiate the Modal App
app = modal.App("workforce-digital-twin-backend", image=image)

# 2. Asynchronous Background Task Worker
# This function runs completely isolated inside a dedicated cloud container
@app.function()
def run_simulation_task(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Lazy import inside the container boundary ensures clean decoupling
    from engine import LivingMonteCarloSimulator
    
    print(f"Background worker received simulation job. Trials: {config_dict.get('trials')}")
    simulator = LivingMonteCarloSimulator(config=config_dict)
    
    # Execute the heavy core optimization code we verified in Phase 1
    results = simulator.run_simulation()
    return results

# 3. Web API Submission Endpoint
# Streamlit will POST config JSON payloads here. Returns an immediate tracking ID.
@app.web_endpoint(method="POST")
def submit_simulation(config_dict: Dict[str, Any]):
    # .spawn() tells Modal to queue the task and return instantly without waiting
    call = run_simulation_task.spawn(config_dict)
    
    # The call.object_id is a unique cryptographically safe tracker (e.g., fc-xxxx)
    return {
        "status": "PENDING",
        "job_id": call.object_id,
        "message": "Simulation successfully queued asynchronously."
    }

# 4. Web API Polling Endpoint
# Streamlit calls this GET endpoint every few seconds to check execution progress
@app.web_endpoint(method="GET")
def check_status(job_id: str):
    from modal.functions import FunctionCall
    import modal.exception
    
    try:
        # Reconstruct the function call trace using the tracked Job ID
        f_call = FunctionCall.from_id(job_id)
        
        # Non-blocking peek: Try to resolve the container's output with a 100ms timeout
        result = f_call.get(timeout=0.1)
        
        return {
            "status": "SUCCESS",
            "result": result
        }
    except TimeoutError:
        # The container is still computing the MILP/Monte Carlo models
        return {
            "status": "RUNNING",
            "message": "Simulation actively running on backend worker."
        }
    except Exception as e:
        # Catch unexpected computational or constraint failures safely
        return {
            "status": "FAILED",
            "error": str(e)
        }
