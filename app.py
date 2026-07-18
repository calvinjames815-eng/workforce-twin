import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json

# ==========================================
# BACKEND API CONFIGURATION
# ==========================================
# During local testing, change these to http://localhost:8000/...
MODAL_SUBMIT_URL = "https://calvinjames815-eng--workforce-digital-twin-backend-submi-a74549.modal.run"
MODAL_CHECK_URL = "https://calvinjames815-eng--workforce-digital-twin-backend-check-status.modal.run"

st.set_page_config(page_title="Workforce Digital Twin Dashboard", page_icon="👥", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "simulation_results" not in st.session_state: st.session_state["simulation_results"] = None
if "job_id" not in st.session_state: st.session_state["job_id"] = None
if "job_status" not in st.session_state: st.session_state["job_status"] = None
if "job_error" not in st.session_state: st.session_state["job_error"] = None

# ==========================================
# HELPERS
# ==========================================
def parse_backend_results(raw_data: dict) -> dict:
    """Safely reconstructs DataFrames from the backend JSON response."""
    parsed = {}
    record_based_keys = ["allocations", "employees", "projects", "kpis", "burnout", "departments", "project_summary"]
    
    for key in record_based_keys:
        data = raw_data.get(key, [])
        parsed[key] = pd.DataFrame(data) if data else pd.DataFrame()
            
    summary_data = raw_data.get("simulation_summary", {})
    parsed["simulation_summary"] = pd.DataFrame.from_dict(summary_data, orient='index') if summary_data else pd.DataFrame()
    return parsed

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Simulation Configurator")
trials = st.sidebar.slider("Monte Carlo Trials", 5, 200, 50, 5)
steps = st.sidebar.slider("Planning Cycles Per Trial", 4, 24, 12, 1)
employees = st.sidebar.slider("Initial Workforce Size", 20, 10000, 2500, 100)
projects = st.sidebar.slider("Backlog Pipeline Size", 10, 1000, 300, 10)

col_run, col_reset = st.sidebar.columns(2)

with col_run:
    is_running = st.session_state["job_id"] is not None
    if st.button("Run Simulation", type="primary", use_container_width=True, disabled=is_running):
        st.session_state["simulation_results"] = None
        st.session_state["job_error"] = None
        
        payload = {
            "trials": trials, "steps_per_trial": steps,
            "initial_employees": employees, "initial_projects": projects
        }
        
        try:
            response = requests.post(MODAL_SUBMIT_URL, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.session_state["job_id"] = data.get("job_id")
                st.session_state["job_status"] = "PENDING"
                st.rerun()
            else:
                st.session_state["job_error"] = f"Backend error: {response.status_code}"
        except Exception as e:
            st.session_state["job_error"] = f"Dispatch failed: {str(e)}"

with col_reset:
    if st.button("Reset", use_container_width=True):
        for key in ["simulation_results", "job_id", "job_status", "job_error"]:
            st.session_state[key] = None
        st.rerun()

# ==========================================
# ASYNC POLLING STATE MACHINE
# ==========================================
if st.session_state["job_id"]:
    with st.sidebar.status(f"Job: {st.session_state['job_id'][:8]}...", expanded=True) as status_box:
        st.write(f"Status: **{st.session_state['job_status']}**")
        try:
            poll_response = requests.get(f"{MODAL_CHECK_URL}?job_id={st.session_state['job_id']}", timeout=5)
            if poll_response.status_code == 200:
                poll_data = poll_response.json()
                current_status = poll_data.get("status")
                st.session_state["job_status"] = current_status
                
                if current_status == "SUCCESS":
                    status_box.update(label="Complete!", state="complete")
                    st.session_state["simulation_results"] = parse_backend_results(poll_data.get("result", {}))
                    st.session_state["job_id"] = None
                    st.rerun()
                elif current_status == "FAILED":
                    status_box.update(label="Failed", state="error")
                    st.session_state["job_error"] = poll_data.get("error", "Unknown error")
                    st.session_state["job_id"] = None
                else:
                    time.sleep(2) # Prevent rapid-fire polling
                    st.rerun()
            else:
                st.session_state["job_error"] = "Polling error."
                st.session_state["job_id"] = None
        except Exception as e:
            st.session_state["job_error"] = str(e)
            st.session_state["job_id"] = None

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("👥 Enterprise Workforce Digital Twin")

if st.session_state["job_error"]:
    st.error(st.session_state["job_error"])

if st.session_state["simulation_results"]:
    results = st.session_state["simulation_results"]
    # ... (Your visualization code goes here, same as provided in previous snippets)
    st.success("Simulation data loaded.")
else:
    st.info("Select parameters and click 'Run Simulation'.")
