import streamlit as st
import pandas as pd
import requests
import time

MODAL_SUBMIT_URL = st.secrets.get("MODAL_SUBMIT_URL", "https://your-workspace--workforce-digital-twin-backend-submit-simulation.modal.run")
MODAL_CHECK_URL = st.secrets.get("MODAL_CHECK_URL", "https://your-workspace--workforce-digital-twin-backend-check-status.modal.run")
MODAL_RESULT_URL = st.secrets.get("MODAL_RESULT_URL", "https://your-workspace--workforce-digital-twin-backend-get-result.modal.run")

st.set_page_config(page_title="Workforce Digital Twin Dashboard", page_icon="🧩", layout="wide")

API_KEY = st.secrets.get("X_API_KEY") or st.secrets.get("INTERNAL_API_KEY") or st.secrets.get("API_KEY", "")

if not API_KEY:
    st.warning("⚠️ **Missing API Key** — Please set `X_API_KEY` in Streamlit secrets to run simulations.")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

MAX_POLL_RETRIES = 5

# --- SESSION STATE INITIALIZATION ---
if "simulation_results" not in st.session_state: st.session_state["simulation_results"] = None
if "job_id" not in st.session_state: st.session_state["job_id"] = None
if "job_status" not in st.session_state: st.session_state["job_status"] = None
if "job_error" not in st.session_state: st.session_state["job_error"] = None
if "poll_retry_count" not in st.session_state: st.session_state["poll_retry_count"] = 0

def parse_backend_results(raw_data: dict) -> dict:
    parsed = {}
    record_based_keys = ["kpis", "burnout", "departments", "project_summary", "allocation_summary"]
    for key in record_based_keys:
        data = raw_data.get(key, [])
        parsed[key] = pd.DataFrame(data) if data else pd.DataFrame()

    parsed["performance_summary"] = raw_data.get("performance_summary", {})
    return parsed

def _give_up_polling(status_box, label: str, message: str) -> None:
    """Terminate the current job with a clear, user-visible error."""
    status_box.update(label=label, state="error")
    st.session_state["job_error"] = message
    st.session_state["job_id"] = None
    st.session_state["poll_retry_count"] = 0

def _retry_with_backoff(status_box, detail: str) -> None:
    """
    Shared bounded-retry-with-exponential-backoff path, used for BOTH connection-level
    failures (DNS/refused/timeout) and transient server-side errors (5xx). A 5xx from a
    serverless backend (e.g. Modal cold start / container recycle) is just as likely to
    resolve on its own as a network blip, so both are treated as retryable; only 4xx
    client errors (bad job_id, auth failure, etc.) are treated as non-retryable below.
    """
    st.session_state["poll_retry_count"] += 1
    if st.session_state["poll_retry_count"] > MAX_POLL_RETRIES:
        _give_up_polling(
            status_box,
            "Connection Lost",
            f"Lost connection to backend after {MAX_POLL_RETRIES} retry attempts. {detail}".strip()
        )
    else:
        backoff_time = 2 ** st.session_state["poll_retry_count"]
        status_box.update(
            label=f"{detail} Retrying ({st.session_state['poll_retry_count']}/{MAX_POLL_RETRIES})...",
            state="running"
        )
        time.sleep(backoff_time)
        st.rerun()

st.sidebar.header("Simulation Configurator")
trials = st.sidebar.slider("Monte Carlo Trials", 5, 200, 50, 5)
steps = st.sidebar.slider("Planning Cycles Per Trial", 4, 24, 12, 1)
employees = st.sidebar.slider("Initial Workforce Size", 20, 5000, 100, 10)
projects = st.sidebar.slider("Backlog Pipeline Size", 10, 1000, 40, 5)

col_run, col_reset = st.sidebar.columns(2)

with col_run:
    is_missing_key = not bool(API_KEY)
    is_running = st.session_state["job_id"] is not None

    if st.button("Run Simulation", type="primary", use_container_width=True, disabled=is_running or is_missing_key):
        st.session_state["simulation_results"] = None
        st.session_state["job_error"] = None
        st.session_state["poll_retry_count"] = 0
        payload = {"trials": trials, "steps_per_trial": steps, "initial_employees": employees, "initial_projects": projects}
        try:
            res = requests.post(MODAL_SUBMIT_URL, json=payload, headers=HEADERS, timeout=10)
            if res.status_code == 200:
                st.session_state["job_id"] = res.json().get("job_id")
                st.session_state["job_status"] = "PENDING"
                st.rerun()
            elif res.status_code == 429:
                st.session_state["job_error"] = "Rate limit exceeded. Please wait a minute before submitting again."
            else:
                st.session_state["job_error"] = f"Backend error: {res.status_code} - {res.text[:300]}"
        except Exception as e:
            st.session_state["job_error"] = f"Dispatch failed: {str(e)}"

with col_reset:
    if st.button("Reset", use_container_width=True):
        for key in ["simulation_results", "job_id", "job_status", "job_error"]:
            st.session_state[key] = None
        st.session_state["poll_retry_count"] = 0
        st.rerun()

# --- ROBUST POLLING: bounded retries + backoff shared by network AND 5xx failures ---
if st.session_state["job_id"]:
    with st.sidebar.status(f"Job: {st.session_state['job_id'][:8]}...", expanded=True) as status_box:
        try:
            poll_res = requests.get(f"{MODAL_CHECK_URL}?job_id={st.session_state['job_id']}", headers=HEADERS, timeout=5)

            if poll_res.status_code == 200:
                st.session_state["poll_retry_count"] = 0
                poll_data = poll_res.json()
                status = poll_data.get("status")
                st.session_state["job_status"] = status

                if status == "SUCCESS":
                    status_box.update(label="Downloading Payload...", state="running")
                    dl_res = requests.get(f"{MODAL_RESULT_URL}?result_key={poll_data.get('result_key')}", headers=HEADERS, timeout=120)
                    if dl_res.status_code == 200:
                        status_box.update(label="Complete!", state="complete")
                        st.session_state["simulation_results"] = parse_backend_results(dl_res.json())
                    else:
                        st.session_state["job_error"] = f"Download failed: {dl_res.status_code}"
                    st.session_state["job_id"] = None
                    st.rerun()
                elif status == "FAILED":
                    st.session_state["job_error"] = poll_data.get("error", "Execution failed")
                    st.session_state["job_id"] = None
                else:
                    time.sleep(2)
                    st.rerun()

            elif poll_res.status_code >= 500:
                # Transient server-side error -- treat like a network hiccup, not a hard failure.
                _retry_with_backoff(status_box, f"Server returned HTTP {poll_res.status_code}.")

            else:
                # Non-retryable client-side error (401/404/etc.) -- won't resolve by retrying.
                _give_up_polling(
                    status_box,
                    "Polling Error",
                    f"Status check failed with HTTP {poll_res.status_code}: {poll_res.text[:300]}"
                )

        except requests.exceptions.RequestException:
            _retry_with_backoff(status_box, "Network hiccup.")

        except Exception as e:
            _give_up_polling(status_box, "Fatal Error", str(e))

# --- MAIN DASHBOARD INTERFACE ---
st.title("🧩 Enterprise Workforce Digital Twin")

if st.session_state["job_error"]:
    st.error(st.session_state["job_error"])

if st.session_state["simulation_results"]:
    results = st.session_state["simulation_results"]
    kpis = results["kpis"]
    allocations = results["allocation_summary"]  # trial+step+role aggregate; see backend.py note
    burnout = results["burnout"]
    departments = results["departments"]
    project_summary = results["project_summary"]
    perf_summary = results["performance_summary"]

    st.success("Simulation data loaded.")

    if not kpis.empty:
        kpis_mean = kpis.groupby("Step").mean(numeric_only=True).reset_index()
        latest = kpis_mean.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Utilization (%)", round(latest.get("Utilization (%)", 0), 2))
        c2.metric("Avg Projects Completed", round(latest.get("Projects Completed", 0), 1))
        c3.metric("Avg Burnout", round(latest.get("Avg Burnout", 0), 3))
        c4.metric("Resource Contention", round(latest.get("Resource Contention", 0), 3))

    tab_kpis, tab_projects, tab_workforce, tab_perf = st.tabs(["📈 KPI Trends", "📂 Projects", "👥 Workforce", "⚙️ Performance"])

    with tab_kpis:
        if not kpis.empty:
            kpis_indexed = kpis_mean.set_index("Step")
            st.line_chart(kpis_indexed[["Utilization (%)", "Avg Burnout", "Resource Contention"]])
            st.line_chart(kpis_indexed[["Projects Completed", "Projects Failed/Shelved", "Active Backlog Size"]])
            with st.expander("Raw KPI table"):
                st.dataframe(kpis, use_container_width=True)

    with tab_projects:
        if not project_summary.empty:
            st.bar_chart(project_summary.set_index("status")["count"])
        with st.expander("Allocation Summary Table (by trial / step / role)"):
            st.dataframe(allocations, use_container_width=True)

    with tab_workforce:
        if not departments.empty:
            st.subheader("Department / Role Summary")
            st.dataframe(departments, use_container_width=True)
        if not burnout.empty:
            st.subheader("Burnout Trajectory Across Planning Steps")
            b_indexed = burnout.set_index("step") if "step" in burnout.columns else burnout
            st.line_chart(b_indexed["avg_rolling_fatigue"])

    with tab_perf:
        if perf_summary:
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Sim Time (s)", perf_summary.get("total_simulation_time"))
            p2.metric("Solver % of Runtime", f'{perf_summary.get("solver_percentage_of_total_runtime")}%')
            p3.metric("Avg Solve Time/Cycle (s)", perf_summary.get("avg_solve_time_per_cycle"))
            st.json(perf_summary)
else:
    st.info("Select parameters and click 'Run Simulation'.")
