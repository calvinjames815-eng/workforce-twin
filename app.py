import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

# ==========================================
# BACKEND API CONFIGURATION
# ==========================================
# Replace these strings with your exact permanent public URLs generated during 'modal deploy'
MODAL_SUBMIT_URL = "https://YOUR_MODAL_USERNAME--workforce-digital-twin-backend-submit-simulation.modal.run"
MODAL_CHECK_URL = "https://YOUR_MODAL_USERNAME--workforce-digital-twin-backend-check-status.modal.run"

# Set page configuration for enterprise dashboard styling
st.set_page_config(
    page_title="Workforce Digital Twin Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State tracking parameters
if "simulation_results" not in st.session_state:
    st.session_state["simulation_results"] = None
if "job_id" not in st.session_state:
    st.session_state["job_id"] = None
if "job_status" not in st.session_state:
    st.session_state["job_status"] = None
if "job_error" not in st.session_state:
    st.session_state["job_error"] = None

# Helper function to reconstruct JSON objects back into local operational DataFrames
def parse_backend_results(raw_data: dict) -> dict:
    parsed = {}
    record_based_keys = [
        "allocations", "employees", "projects", "kpis", 
        "burnout", "departments", "project_summary"
    ]
    for key in record_based_keys:
        if key in raw_data:
            parsed[key] = pd.DataFrame(raw_data[key])
        else:
            parsed[key] = pd.DataFrame()
            
    if "simulation_summary" in raw_data:
        parsed["simulation_summary"] = pd.DataFrame.from_dict(raw_data["simulation_summary"], orient='index')
    else:
        parsed["simulation_summary"] = pd.DataFrame()
        
    return parsed

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Simulation Configurator")

# Simulation Parameters
trials = st.sidebar.slider("Monte Carlo Trials", min_value=5, max_value=200, value=50, step=5)
steps = st.sidebar.slider("Planning Cycles Per Trial", min_value=4, max_value=24, value=12, step=1)
employees = st.sidebar.slider("Initial Workforce Size", min_value=20, max_value=10000, value=2500, step=100)
projects = st.sidebar.slider("Backlog Pipeline Size", min_value=10, max_value=1000, value=300, step=10)

st.sidebar.markdown("---")
st.sidebar.header("Dashboard Settings")
show_raw_tables = st.sidebar.toggle("Show Raw Data Tables Below Visuals", value=False)

st.sidebar.caption("💡 Tip: Toggle Dark/Light mode in the top-right Streamlit settings menu.")

# Simulation Trigger Buttons
col_run, col_reset = st.sidebar.columns(2)

with col_run:
    # Disable run button if a simulation is actively executing in the background
    is_running = st.session_state["job_id"] is not None
    if st.button("Run Simulation", type="primary", use_container_width=True, disabled=is_running):
        # Reset any previous session parameters before dispatching network tasks
        st.session_state["simulation_results"] = None
        st.session_state["job_error"] = None
        
        # Package raw slider parameters into web-safe JSON configurations
        payload = {
            "trials": trials,
            "steps_per_trial": steps,
            "initial_employees": employees,
            "initial_projects": projects
        }
        
        try:
            # Asynchronously submit request to the cloud engine infrastructure
            response = requests.post(MODAL_SUBMIT_URL, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                st.session_state["job_id"] = data.get("job_id")
                st.session_state["job_status"] = "PENDING"
                st.rerun()
            else:
                st.session_state["job_error"] = f"Backend API returned status code {response.status_code}."
        except Exception as e:
            st.session_state["job_error"] = f"Failed to dispatch job to serverless engine: {str(e)}"

with col_reset:
    if st.button("Reset", type="secondary", use_container_width=True):
        st.session_state["simulation_results"] = None
        st.session_state["job_id"] = None
        st.session_state["job_status"] = None
        st.session_state["job_error"] = None
        st.rerun()

# ==========================================
# ASYNCHRONOUS STATE MACHINE MONITOR
# ==========================================
# This global block acts as a state controller if a background computation job exists
if st.session_state["job_id"]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Worker Node Telemetry")
    
    with st.sidebar.status(f"Job: {st.session_state['job_id'][:8]}...", expanded=True) as status_box:
        st.write(f"Cluster Status: **{st.session_state['job_status']}**")
        
        try:
            # Poll the check_status endpoint via a secure query parameter string
            poll_url = f"{MODAL_CHECK_URL}?job_id={st.session_state['job_id']}"
            poll_response = requests.get(poll_url, timeout=5)
            
            if poll_response.status_code == 200:
                poll_data = poll_response.json()
                current_status = poll_data.get("status")
                st.session_state["job_status"] = current_status
                
                if current_status == "SUCCESS":
                    status_box.update(label="✓ Core Calculations Complete!", state="complete", expanded=False)
                    # Unpack and rebuild DataFrames
                    st.session_state["simulation_results"] = parse_backend_results(poll_data.get("result", {}))
                    st.session_state["job_id"] = None  # Clear job identifier to unlock layout interactions
                    st.rerun()
                    
                elif current_status == "FAILED":
                    status_box.update(label="✕ Distributed Task Failed", state="error", expanded=True)
                    st.session_state["job_error"] = poll_data.get("error", "Unknown pipeline error.")
                    st.session_state["job_id"] = None
                    
                else:
                    # Job state is PENDING or RUNNING. Sleep briefly and re-trigger layout pass
                    time.sleep(2.5)
                    st.rerun()
            else:
                st.session_state["job_error"] = f"Network telemetry error. Status code: {poll_response.status_code}"
                st.session_state["job_id"] = None
                st.rerun()
                
        except Exception as e:
            st.session_state["job_error"] = f"Telemetry polling exception: {str(e)}"
            st.session_state["job_id"] = None
            st.rerun()

# ==========================================
# MAIN DASHBOARD INTERFACE
# ==========================================
st.title("👥 Enterprise Workforce Digital Twin")
st.markdown("Mathematical mixed-integer resource allocation, career path progression, and burnout risk modeling.")

# Visual status notifications for error reporting
if st.session_state["job_error"]:
    st.error(st.session_state["job_error"])

# Active background workload progress spinner
if st.session_state["job_id"]:
    st.info(f"⏳ **Active Serverless Compute:** Running optimization matrix models via distributed workers. Please wait...")
    st.spinner("Crunching mixed integer linear programming models across Monte Carlo instances...")

# Verify if data exists; if not, show placeholder layout
if st.session_state["simulation_results"] is None:
    if not st.session_state["job_id"]:
        st.info("👈 Please select your workforce constraints in the sidebar and click 'Run Simulation' to generate analytics.")
    
    st.subheader("Interactive Module Blueprint")
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80",
        caption="Ready to ingest allocations, employee state vectors, project backlogs, and step-by-step KPIs.",
        use_container_width=True
    )
else:
    results = st.session_state["simulation_results"]
    kpis_df = results.get("kpis", pd.DataFrame())
    allocs_df = results.get("allocations", pd.DataFrame())
    emps_df = results.get("employees", pd.DataFrame())
    projs_df = results.get("projects", pd.DataFrame())
    burnout_df = results.get("burnout", pd.DataFrame())
    depts_df = results.get("departments", pd.DataFrame())
    proj_summary = results.get("project_summary", pd.DataFrame())
    sim_summary = results.get("simulation_summary", pd.DataFrame())

    # ==========================================
    # OVERVIEW STATUS MATRIX
    # ==========================================
    st.subheader("Global Portfolio Health Matrix")
    
    avg_util = kpis_df["Utilization (%)"].mean() if (not kpis_df.empty and "Utilization (%)" in kpis_df.columns) else 0.0
    avg_burn = kpis_df["Avg Burnout"].mean() if (not kpis_df.empty and "Avg Burnout" in kpis_df.columns) else 0.0
    avg_cont = kpis_df["Resource Contention"].mean() if (not kpis_df.empty and "Resource Contention" in kpis_df.columns) else 0.0
    
    if not kpis_df.empty and "Trial" in kpis_df.columns:
        completed_total = kpis_df.groupby("Trial")["Projects Completed"].max().mean() if "Projects Completed" in kpis_df.columns else 0.0
        failed_total = kpis_df.groupby("Trial")["Projects Failed/Shelved"].max().mean() if "Projects Failed/Shelved" in kpis_df.columns else 0.0
        final_backlog = kpis_df.groupby("Trial")["Active Backlog Size"].last().mean() if "Active Backlog Size" in kpis_df.columns else 0.0
    else:
        completed_total, failed_total, final_backlog = 0.0, 0.0, 0.0

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    m_col1.metric(label="Avg Capacity Utilization", value=f"{avg_util:.1f}%")
    m_col2.metric(label="Avg Burnout Index", value=f"{avg_burn:.2f}", help="Scale of 0.0 (Fresh) to 2.0 (Exhausted)")
    m_col3.metric(label="Contention Index", value=f"{avg_cont:.2f}", help="Active projects per available employee")
    m_col4.metric(label="Avg Projects Completed", value=f"{completed_total:.1f}")
    m_col5.metric(label="Avg Projects Failed/Shelved", value=f"{failed_total:.1f}")
    m_col6.metric(label="Avg Final Backlog Queue", value=f"{final_backlog:.1f}")

    st.markdown("---")

    # ==========================================
    # WORKSPACE TABBED VIEWPORT
    # ==========================================
    tab_kpi, tab_workforce, tab_projects, tab_allocations, tab_departments, tab_burnout, tab_summary = st.tabs([
        "📈 KPI Trends", 
        "👥 Workforce Demographics", 
        "📁 Project Backlog Status", 
        "🛠️ Optimized Allocations", 
        "🏢 Department Metrics", 
        "🔥 Burnout Analytics", 
        "📊 Simulation Statistics"
    ])

    # ------------------------------------------
    # TAB 1: KPI TRENDS
    # ------------------------------------------
    with tab_kpi:
        st.subheader("Portfolio Performance Progression over Time")
        st.markdown("Metrics are aggregated and averaged step-by-step across all Monte Carlo simulation trials.")

        if not kpis_df.empty and "Step" in kpis_df.columns:
            step_trends = kpis_df.groupby("Step").mean(numeric_only=True).reset_index()

            col_trend1, col_trend2 = st.columns(2)
            with col_trend1:
                st.markdown("**Capacity Utilization & Burnout Trend**")
                cols1 = [c for c in ["Utilization (%)", "Avg Burnout"] if c in step_trends.columns]
                if cols1:
                    st.line_chart(step_trends.set_index("Step")[cols1], use_container_width=True)
                else:
                    st.info("Required metric columns for utilization or burnout trend are unavailable.")
                
                st.markdown("**Backlog Size & Resource Contention Pressure**")
                cols2 = [c for c in ["Active Backlog Size", "Resource Contention"] if c in step_trends.columns]
                if cols2:
                    st.line_chart(step_trends.set_index("Step")[cols2], use_container_width=True)
                else:
                    st.info("Required metric columns for backlog size or resource contention pressure are unavailable.")

            with col_trend2:
                st.markdown("**Project Completions & Failure Casings**")
                cols3 = [c for c in ["Projects Completed", "Projects Failed/Shelved"] if c in step_trends.columns]
                if cols3:
                    st.line_chart(step_trends.set_index("Step")[cols3], use_container_width=True)
                else:
                    st.info("Required metric columns for project completions or failures are unavailable.")

                st.markdown("**Average Structural Project Delay (Staffing Factor)**")
                if "Avg Project Delay (Factor)" in step_trends.columns:
                    st.line_chart(step_trends.set_index("Step")[["Avg Project Delay (Factor)"]], use_container_width=True)
                else:
                    st.info("Average project delay columns are unavailable.")

            if show_raw_tables:
                st.subheader("Step-by-Step Trend Data")
                st.dataframe(step_trends, use_container_width=True)
        else:
            st.info("No KPI data available to display trends.")

    # ------------------------------------------
    # TAB 2: WORKFORCE DEMOGRAPHICS
    # ------------------------------------------
    with tab_workforce:
        st.subheader("Worker State Profile (Representative Terminal State)")
        
        if not emps_df.empty:
            w_col1, w_col2, w_col3 = st.columns(3)
            with w_col1:
                st.markdown("**Workforce Count by Primary Role**")
                if "role" in emps_df.columns:
                    role_counts = emps_df["role"].value_counts().reset_index()
                    role_counts.columns = ["Role", "Employee Count"]
                    st.bar_chart(role_counts.set_index("Role"), use_container_width=True)
                else:
                    st.info("Role metadata is absent from employee schema.")

                st.markdown("**Seniority Level Distribution**")
                if "seniority" in emps_df.columns:
                    seniority_hist = np.histogram(emps_df["seniority"].dropna(), bins=10)
                    hist_df_s = pd.DataFrame({"Seniority Density": seniority_hist[0]}, index=seniority_hist[1][:-1])
                    st.bar_chart(hist_df_s, use_container_width=True)
                else:
                    st.info("Seniority metric is absent from employee data.")

            with w_col2:
                st.markdown("**Work-Based Experience Distribution**")
                if "experience" in emps_df.columns:
                    exp_hist = np.histogram(emps_df["experience"].dropna(), bins=10)
                    hist_df_e = pd.DataFrame({"Experience Level Density": exp_hist[0]}, index=exp_hist[1][:-1])
                    st.bar_chart(hist_df_e, use_container_width=True)
                else:
                    st.info("Experience metric is absent from employee data.")

                st.markdown("**Current Productive Capacity (Efficiency) Distribution**")
                if "efficiency" in emps_df.columns:
                    eff_hist = np.histogram(emps_df["efficiency"].dropna(), bins=10)
                    hist_df_eff = pd.DataFrame({"Worker Efficiency Density": eff_hist[0]}, index=eff_hist[1][:-1])
                    st.bar_chart(hist_df_eff, use_container_width=True)
                else:
                    st.info("Efficiency metric is absent from employee data.")

            with w_col3:
                st.markdown("**Elastic Concurrency Allocation Limits**")
                if "concurrency_limit" in emps_df.columns:
                    con_counts = emps_df["concurrency_limit"].value_counts().reset_index()
                    con_counts.columns = ["Maximum Allowed Projects", "Count"]
                    st.bar_chart(con_counts.set_index("Maximum Allowed Projects"), use_container_width=True)
                else:
                    st.info("Elastic concurrency profiles are absent or disabled for this simulation step.")
                
                st.markdown("**Worker Statistics Summary**")
                st.dataframe(emps_df.describe().T, use_container_width=True)
        else:
            st.info("No workforce demographic data available.")

    # ------------------------------------------
    # TAB 3: PROJECT BACKLOG STATUS
    # ------------------------------------------
    with tab_projects:
        st.subheader("Project Inventory Pipeline")
        
        if not projs_df.empty:
            p_col1, p_col2 = st.columns([1, 2])
            with p_col1:
                st.markdown("**Aggregated Lifecycle Statuses**")
                if not proj_summary.empty:
                    st.dataframe(proj_summary, use_container_width=True)
                else:
                    st.info("No categorical execution summary calculated.")

                st.markdown("**Prerequisite System Audit**")
                has_dep = projs_df["prerequisite_id"].notna().sum() if "prerequisite_id" in projs_df.columns else 0
                blocked_count = projs_df["is_blocked"].sum() if "is_blocked" in projs_df.columns else 0
                st.write(f"Total Projects bound by Dependencies: **{has_dep}**")
                st.write(f"Current Blocked/Prerequisite-waiting Projects: **{blocked_count}**")

            with p_col2:
                st.markdown("**Backlog Density by Priority and Required Role**")
                p_cols = [c for c in ["required_role", "priority"] if c in projs_df.columns]
                if len(p_cols) == 2:
                    backlog_matrix = projs_df.groupby(p_cols).size().unstack(fill_value=0)
                    st.bar_chart(backlog_matrix, use_container_width=True)
                else:
                    st.info("Required matrix columns for categorization are unavailable.")

            st.markdown("**Current Active Backlog Queue**")
            if "status" in projs_df.columns:
                p_status_filter = st.selectbox("Filter Projects by Status", options=["All"] + list(projs_df["status"].dropna().unique()))
                display_projs = projs_df if p_status_filter == "All" else projs_df[projs_df["status"] == p_status_filter]
            else:
                display_projs = projs_df
            
            st.dataframe(display_projs, use_container_width=True)
        else:
            st.info("No project backlog data available.")

    # ------------------------------------------
    # TAB 4: OPTIMIZED ALLOCATIONS
    # ------------------------------------------
    with tab_allocations:
        st.subheader("Hierarchical MILP Assignment Outputs")
        st.markdown("Search, sort, and slice exact optimizer allocation records.")

        if not allocs_df.empty:
            search_query = st.text_input("Search by Employee ID or Project ID:")
            
            role_options = allocs_df["role"].dropna().unique() if "role" in allocs_df.columns else []
            role_filter = st.multiselect("Filter by Assigned Worker Role", options=role_options, default=role_options)
            
            filtered_allocs = allocs_df
            if search_query:
                emp_match = filtered_allocs["emp_id"].str.contains(search_query, case=False, na=False) if "emp_id" in filtered_allocs.columns else False
                proj_match = filtered_allocs["project_id"].str.contains(search_query, case=False, na=False) if "project_id" in filtered_allocs.columns else False
                filtered_allocs = filtered_allocs[emp_match | proj_match]
                
            if "role" in filtered_allocs.columns and len(role_options) > 0:
                filtered_allocs = filtered_allocs[filtered_allocs["role"].isin(role_filter)]
            
            st.dataframe(filtered_allocs, use_container_width=True)
        else:
            st.info("No optimizer assignment allocations were captured for this setup.")

    # ------------------------------------------
    # TAB 5: DEPARTMENT METRICS
    # ------------------------------------------
    with tab_departments:
        st.subheader("Departmental Allocation & Output Averages")

        if not depts_df.empty:
            if "role" in depts_df.columns:
                dept_averages = depts_df.groupby("role").mean(numeric_only=True).reset_index()
                
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.markdown("**Total Allocated Hours by Department**")
                    if "total_hours_allocated" in dept_averages.columns:
                        st.bar_chart(dept_averages.set_index("role")[["total_hours_allocated"]], use_container_width=True)
                    else:
                        st.info("Allocated hour counts are unavailable.")

                    st.markdown("**Average Structural Output Delivered (Hours Adjusted for Efficiency)**")
                    if "avg_output_delivered" in dept_averages.columns:
                        st.bar_chart(dept_averages.set_index("role")[["avg_output_delivered"]], use_container_width=True)
                    else:
                        st.info("Productive delivery hours are unavailable.")

                with d_col2:
                    st.markdown("**Average Role-Match Accuracy Rate**")
                    if "role_match_rate" in dept_averages.columns:
                        st.bar_chart(dept_averages.set_index("role")[["role_match_rate"]], use_container_width=True)
                    else:
                        st.info("Role matching tracking is unavailable.")
                    
                    st.markdown("**Departmental Raw Performance Summary**")
                    st.dataframe(dept_averages, use_container_width=True)
            else:
                st.info("Department configuration parameters are mismatching.")
        else:
            st.info("No department allocation summaries could be extracted from the output matrix.")

    # ------------------------------------------
    # TAB 6: BURNOUT ANALYTICS
    # ------------------------------------------
    with tab_burnout:
        st.subheader("Cognitive Load & Worker Exhaustion Vector Auditing")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("**Fatigue Levels Grouped by Steps (Cycle Profile)**")
            if not burnout_df.empty and "step" in burnout_df.columns and "rolling_fatigue" in burnout_df.columns:
                step_burnout = burnout_df.groupby("step")["rolling_fatigue"].mean().reset_index()
                st.line_chart(step_burnout.set_index("step"), use_container_width=True)
            else:
                st.info("Burnout trend data over step indices is unavailable.")

        with b_col2:
            st.markdown("**Workforce Stress Distribution (Fatigue Histogram)**")
            if not emps_df.empty and "rolling_fatigue" in emps_df.columns:
                fatigue_hist = np.histogram(emps_df["rolling_fatigue"].dropna(), bins=10)
                hist_df_b = pd.DataFrame({"Burnout Level Density": fatigue_hist[0]}, index=fatigue_hist[1][:-1])
                st.bar_chart(hist_df_b, use_container_width=True)
            else:
                st.info("Exhaustion density variables are missing.")

        st.markdown("**Highest Attrition/Burnout Risk Standouts (Fatigue > 1.2)**")
        if not emps_df.empty and "rolling_fatigue" in emps_df.columns:
            high_risk_emps = emps_df[emps_df["rolling_fatigue"] > 1.2].sort_values(by="rolling_fatigue", ascending=False)
            if not high_risk_emps.empty:
                st.dataframe(high_risk_emps, use_container_width=True)
            else:
                st.success("Excellent! No employees are currently in the high burnout danger zone (>1.2 fatigue index).")
        else:
            st.info("Worker capacity logs are empty.")

    # ------------------------------------------
    # TAB 7: SIMULATION STATISTICS
    # ------------------------------------------
    with tab_summary:
        st.subheader("Monte Carlo Trial Statistical Distribution")
        st.markdown("A comprehensive describe summary of critical KPI ranges across every executed planning step.")
        if not sim_summary.empty:
            st.dataframe(sim_summary, use_container_width=True)
        else:
            st.info("Statistical metrics are missing or unavailable.")

    # ==========================================
    # GLOBAL EXPORTS PANEL
    # ==========================================
    st.markdown("---")
    st.subheader("📥 Export Digital Twin Analytics")
    st.markdown("Download simulation run logs to persist results for downstream reporting.")

    dl_col1, dl_col2, dl_col3, dl_col4, dl_col5, dl_col6 = st.columns(6)
    
    to_csv_bytes = lambda df: df.to_csv(index=False).encode('utf-8') if not df.empty else b""

    with dl_col1:
        st.download_button(
            label="Download Allocations",
            data=to_csv_bytes(allocs_df),
            file_name="sim_allocations.csv",
            mime="text/csv",
            disabled=allocs_df.empty,
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="Download Workforce State",
            data=to_csv_bytes(emps_df),
            file_name="sim_employees.csv",
            mime="text/csv",
            disabled=emps_df.empty,
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            label="Download Backlog Data",
            data=to_csv_bytes(projs_df),
            file_name="sim_projects.csv",
            mime="text/csv",
            disabled=projs_df.empty,
            use_container_width=True
        )
    with dl_col4:
        st.download_button(
            label="Download Step KPIs",
            data=to_csv_bytes(kpis_df),
            file_name="sim_kpis.csv",
            mime="text/csv",
            disabled=kpis_df.empty,
            use_container_width=True
        )
    with dl_col5:
        st.download_button(
            label="Download Burnout Logs",
            data=to_csv_bytes(burnout_df),
            file_name="sim_burnouts.csv",
            mime="text/csv",
            disabled=burnout_df.empty,
            use_container_width=True
        )
    with dl_col6:
        st.download_button(
            label="Download Department KPIs",
            data=to_csv_bytes(depts_df),
            file_name="sim_departments.csv",
            mime="text/csv",
            disabled=depts_df.empty,
            use_container_width=True
        )
