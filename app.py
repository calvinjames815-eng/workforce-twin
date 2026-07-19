import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json

# ==========================================
# BACKEND API CONFIGURATION
# ==========================================
# Preserved existing URLs and appended standard structure for the new GET endpoint
MODAL_SUBMIT_URL = "https://calvinjames815-eng--workforce-digital-twin-backend-submi-a74549.modal.run"
MODAL_CHECK_URL = "https://calvinjames815-eng--workforce-digital-twin-backend-check-status.modal.run"
MODAL_RESULT_URL = "https://calvinjames815-eng--workforce-digital-twin-backend-get-result.modal.run"

st.set_page_config(page_title="Workforce Digital Twin Dashboard", page_icon="🧩", layout="wide")

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
    parsed = {}
    # Strictly pull the pre-aggregated payload layout keys
    record_based_keys = ["allocations", "kpis", "burnout", "departments", "project_summary"]
    
for key in record_based_keys:
        data = raw_data.get(key, [])
        parsed[key] = pd.DataFrame(data) if data else pd.DataFrame()
       
    parsed["performance_summary"] = raw_data.get("performance_summary", {})
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
                    status_box.update(label="Downloading Payload...", state="running")
                    
                    # Target endpoint download containing optimized 120 second network timeout
                    res_key = poll_data.get("result_key")
                    download_response = requests.get(f"{MODAL_RESULT_URL}?result_key={res_key}", timeout=120)
                    
                    if download_response.status_code == 200:
                        status_box.update(label="Complete!", state="complete")
                        st.session_state["simulation_results"] = parse_backend_results(download_response.json())
                    else:
                        status_box.update(label="Download Error", state="error")
                        st.session_state["job_error"] = f"Failed to acquire results file: {download_response.status_code} - {download_response.text}"
                        
                    st.session_state["job_id"] = None
                    st.rerun()
                elif current_status == "FAILED":
                    status_box.update(label="Failed", state="error")
                    st.session_state["job_error"] = poll_data.get("error", "Unknown error encountered")
                    st.session_state["job_id"] = None
                else:
                    time.sleep(2)
                    st.rerun()
            else:
                st.session_state["job_error"] = "Polling transport failure."
                st.session_state["job_id"] = None
        except Exception as e:
            st.session_state["job_error"] = str(e)
            st.session_state["job_id"] = None

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("🧩 Enterprise Workforce Digital Twin")

if st.session_state["job_error"]:
    st.error(st.session_state["job_error"])

if st.session_state["simulation_results"]:
    results = st.session_state["simulation_results"]

    kpis = results["kpis"]
    allocations = results["allocations"]
    burnout = results["burnout"]
    departments = results["departments"]
    project_summary = results["project_summary"]
    perf_summary = results["performance_summary"]

    st.success("Simulation data loaded.")

    # ---- Top-line KPI cards (latest cycle) ----
    if not kpis.empty:
        latest = kpis.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Utilization (%)", latest["Utilization (%)"])
        c2.metric("Projects Completed", int(latest["Projects Completed"]))
        c3.metric("Avg Burnout", latest["Avg Burnout"])
        c4.metric("Resource Contention", latest["Resource Contention"])

    tab_kpis, tab_projects, tab_workforce, tab_perf = st.tabs(
        ["📈 KPI Trends", "📂 Projects", "👥 Workforce", "⚙️ Solver Performance"]
    )

    with tab_kpis:
        if not kpis.empty:
            kpis_indexed = kpis.reset_index(drop=True)
            kpis_indexed["cycle"] = range(len(kpis_indexed))
            st.line_chart(kpis_indexed.set_index("cycle")[["Utilization (%)", "Avg Burnout", "Resource Contention"]])
            st.line_chart(kpis_indexed.set_index("cycle")[["Projects Completed", "Projects Failed/Shelved", "Active Backlog Size"]])
            with st.expander("Raw KPI table"):
                st.dataframe(kpis, use_container_width=True)
        else:
            st.info("No KPI data returned.")

    with tab_projects:
        if not project_summary.empty:
            st.bar_chart(project_summary.set_index("status")["count"])
        with st.expander("Aggregated Allocations Table"):
            st.dataframe(allocations, use_container_width=True)

    with tab_workforce:
        if not departments.empty:
            st.subheader("Department / Role Summary")
            st.dataframe(departments, use_container_width=True)
        if not burnout.empty:
            st.subheader("Burnout by cycle")
            # Directly plot without secondary frontend grouping
            burnout_display = burnout.copy()
            burnout_display["cycle"] = range(len(burnout_display))
            st.line_chart(burnout_display.set_index("cycle")["avg_rolling_fatigue"])
        with st.expander("Aggregated Burnout Table"):
            st.dataframe(burnout, use_container_width=True)

    with tab_perf:
        if perf_summary:
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Sim Time (s)", perf_summary.get("total_simulation_time"))
            p2.metric("Solver % of Runtime", f'{perf_summary.get("solver_percentage_of_total_runtime")}%')
            p3.metric("Avg Solve Time/Cycle (s)", perf_summary.get("avg_solve_time_per_cycle"))
            st.json(perf_summary)
        else:
            st.info("No performance telemetry returned.")

else:
    st.info("Select parameters and click 'Run Simulation'.")
