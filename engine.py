import json
import sys
import time
import numpy as np
import pandas as pd
import pulp
from dataclasses import dataclass, fields, asdict
from typing import Dict, List, Tuple, Any, Optional

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
BASE_HOURS_PER_PROJECT = 20
MAX_CONCURRENT_PROJECTS = 2
STANDARD_WEEKLY_HOURS = 40
OVERTIME_MAX_HOURS = 48
CROSS_ROLE_PENALTY = 0.5
PARTIAL_STAFF_PENALTY = 0.8

@dataclass
class SimulationConfig:
    trials: int = 100
    steps_per_trial: int = 12
    initial_employees: int = 100
    initial_projects: int = 40
    new_projects_per_step: int = 5
    hiring_interval_steps: int = 3
    hires_per_batch: int = 4
    absence_alpha: float = 5.0
    absence_beta: float = 95.0
    retirement_threshold: float = 0.85
    retirement_rate: float = 0.01
    experience_growth_rate: float = 0.05
    promotion_threshold: float = 5.0
    top_k_multiplier: int = 4 

    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationConfig':
        """Safely instantiates configuration from a raw dictionary, filtering out unexpected keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

# ==========================================
# 1. OPTIMIZATION ENGINE (INSTRUMENTED MILP)
# ==========================================
class WorkforceOptimizer:
    """
    Enterprise sparse MILP Optimizer instrumented with high-resolution 
    performance diagnostics and vectorized candidate generation.
    """
    def __init__(self, employees_df: pd.DataFrame, backlog_df: pd.DataFrame, top_k_multiplier: int = 4):
        self.employees_df = employees_df.copy()
        self.backlog_df = backlog_df.copy()
        self.top_k_multiplier = top_k_multiplier
        self.metrics: Dict[str, Any] = {}

    def run_allocation(self, trial_seed: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        t_total_start = time.perf_counter()
        
        if self.employees_df.empty or self.backlog_df.empty:
            self.metrics = {"time_total_optimizer": time.perf_counter() - t_total_start}
            return pd.DataFrame(), []

        df_proj = self.backlog_df.copy()
        df_proj['score'] = df_proj['priority'] * df_proj['urgency']
        df_proj = df_proj.sort_values(by='score', ascending=False)

        emp_ids = self.employees_df['emp_id'].tolist()
        proj_ids = df_proj['project_id'].tolist()

        emp_roles = self.employees_df.set_index('emp_id')['role'].to_dict()
        emp_avails = self.employees_df.set_index('emp_id')['availability'].to_dict()
        emp_effs = self.employees_df.set_index('emp_id')['efficiency'].to_dict()
        emp_fatigue = self.employees_df.set_index('emp_id')['rolling_fatigue'].to_dict()

        proj_roles = df_proj.set_index('project_id')['required_role'].to_dict()
        proj_min_staff = df_proj.set_index('project_id')['min_staff'].to_dict()
        proj_complexity = df_proj.set_index('project_id')['complexity'].to_dict()
        proj_scores = df_proj.set_index('project_id')['score'].to_dict()
        proj_priority = df_proj.set_index('project_id')['priority'].to_dict()

        proj_hours = {
            j: BASE_HOURS_PER_PROJECT + int(10 * (proj_complexity[j] - 1.0)) 
            for j in proj_ids
        }

        # ------------------------------------------
        # METRIC STAGE 1: VECTORIZED CANDIDATE SELECTION & PRUNING
        # ------------------------------------------
        t_cand_start = time.perf_counter()
        
        emp_avails_arr = self.employees_df['availability'].to_numpy()
        emp_roles_arr = self.employees_df['role'].to_numpy()
        emp_effs_arr = self.employees_df['efficiency'].to_numpy()
        emp_fatigue_arr = self.employees_df['rolling_fatigue'].to_numpy()

        proj_roles_arr = df_proj['required_role'].to_numpy()
        proj_min_staff_arr = df_proj['min_staff'].to_numpy()
        proj_priority_arr = df_proj['priority'].to_numpy()
        proj_scores_arr = df_proj['score'].to_numpy()
        proj_complexity_arr = df_proj['complexity'].to_numpy()
        
        proj_hours_arr = BASE_HOURS_PER_PROJECT + (10 * (proj_complexity_arr - 1.0)).astype(int)

        avail_mask = (emp_avails_arr[:, None] > 0) & (emp_avails_arr[:, None] >= proj_hours_arr[None, :])
        role_match_mask = (emp_roles_arr[:, None] == proj_roles_arr[None, :])
        cross_role_allowed = (proj_priority_arr[None, :] >= 3)
        
        eligible_mask = avail_mask & (role_match_mask | cross_role_allowed)

        role_modifier = np.where(role_match_mask, 1.0, CROSS_ROLE_PENALTY)
        fatigue_modifier = 1.0 - (0.1 * emp_fatigue_arr[:, None])
        utility_matrix = proj_scores_arr[None, :] * emp_effs_arr[:, None] * role_modifier * fatigue_modifier

        valid_pairs = []
        utility = {}
        
        for idx, j in enumerate(proj_ids):
            valid_indices = np.where(eligible_mask[:, idx])[0]
            if len(valid_indices) == 0:
                continue
                
            utils = utility_matrix[valid_indices, idx]
            sorted_order = np.argsort(-utils, kind='stable')
            sorted_valid_indices = valid_indices[sorted_order]
            
            max_candidates = proj_min_staff_arr[idx] * self.top_k_multiplier
            truncated_indices = sorted_valid_indices[:max_candidates]
            
            for emp_idx in truncated_indices:
                i = emp_ids[emp_idx]
                valid_pairs.append((i, j))
                utility[i, j] = float(utility_matrix[emp_idx, idx])
                
        t_cand_end = time.perf_counter()

        if not valid_pairs:
            self.metrics = {"time_total_optimizer": time.perf_counter() - t_total_start}
            return pd.DataFrame(), []

        # ------------------------------------------
        # METRIC STAGE 2: ADJACENCY MAP BUILDING
        # ------------------------------------------
        t_adj_start = time.perf_counter()
        emp_to_projs = {i: [] for i in emp_ids}
        proj_to_emps = {j: [] for j in proj_ids}
        
        for (i, j) in valid_pairs:
            emp_to_projs[i].append(j)
            proj_to_emps[j].append(i)
        t_adj_end = time.perf_counter()

        # ------------------------------------------
        # METRIC STAGE 3: MILP CONSTRAINTS FORMULATION
        # ------------------------------------------
        t_build_start = time.perf_counter()
        prob = pulp.LpProblem("Workforce_Allocation_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("assign", valid_pairs, cat=pulp.LpBinary)
        shortfall = pulp.LpVariable.dicts("shortfall", proj_ids, lowBound=0, cat=pulp.LpInteger)

        prob += (
            pulp.lpSum(utility[i, j] * x[i, j] for (i, j) in valid_pairs) - 
            pulp.lpSum(shortfall[j] * (50.0 * proj_scores[j]) for j in proj_ids)
        )

        for i in emp_ids:
            allocated_projects = emp_to_projs[i]
            if allocated_projects:
                prob += pulp.lpSum(proj_hours[j] * x[i, j] for j in allocated_projects) <= emp_avails[i]

        for i in emp_ids:
            allocated_projects = emp_to_projs[i]
            if allocated_projects:
                prob += pulp.lpSum(x[i, j] for j in allocated_projects) <= MAX_CONCURRENT_PROJECTS

        for j in proj_ids:
            eligible_workers = proj_to_emps[j]
            prob += pulp.lpSum(x[i, j] for i in eligible_workers) + shortfall[j] >= proj_min_staff[j]
            prob += pulp.lpSum(x[i, j] for i in eligible_workers) <= (proj_min_staff[j] + 2)
        t_build_end = time.perf_counter()

        # ------------------------------------------
        # METRIC STAGE 4: CBC SOLVER EXECUTION
        # ------------------------------------------
        t_solve_start = time.perf_counter()
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
        status = prob.solve(solver)
        t_solve_end = time.perf_counter()

        # ------------------------------------------
        # METRIC STAGE 5: RESULT MATRIX EXTRACTION
        # ------------------------------------------
        t_extract_start = time.perf_counter()
        assignments = []
        completed_ids = []

        if pulp.LpStatus[status] == "Optimal":
            for j in proj_ids:
                allocated_workers = [i for i in proj_to_emps[j] if x[i, j].varValue == 1]
                staff_count = len(allocated_workers)
                
                if staff_count > 0:
                    needed = proj_min_staff[j]
                    delay_factor = (needed / staff_count) if staff_count > 0 else 2.0
                    eff_penalty = PARTIAL_STAFF_PENALTY if staff_count < needed else 1.0
                    
                    completed_ids.append(j)
                    
                    for i in allocated_workers:
                        allocated_h = proj_hours[j]
                        is_match = (emp_roles[i] == proj_roles[j])
                        role_mod = 1.0 if is_match else CROSS_ROLE_PENALTY
                        fatigue_mod = 1.0 - (0.1 * emp_fatigue[i])
                        
                        effective_h = allocated_h * emp_effs[i] * role_mod * fatigue_mod * eff_penalty
                        
                        assignments.append({
                            'emp_id': i,
                            'project_id': j,
                            'allocated_hours': allocated_h,
                            'effective_output_hours': effective_h,
                            'delay_factor': delay_factor,
                            'role_match': is_match,
                            'priority': proj_priority[j],
                            'role': emp_roles[i]
                        })
        t_extract_end = time.perf_counter()
        t_total_end = time.perf_counter()

        total_employees = len(emp_ids)
        total_projects = len(proj_ids)
        original_search_space = total_employees * total_projects
        pairs_after_pruning = len(valid_pairs)
        reduction_pct = ((original_search_space - pairs_after_pruning) / original_search_space * 100.0) if original_search_space > 0 else 0.0

        self.metrics = {
            "time_candidate_gen_prune": t_cand_end - t_cand_start,
            "time_adjacency_build": t_adj_end - t_adj_start,
            "time_model_build": t_build_end - t_build_start,
            "time_solver_execution": t_solve_end - t_solve_start,
            "time_solution_extraction": t_extract_end - t_extract_start,
            "time_total_optimizer": t_total_end - t_total_start,
            "stat_total_employees": total_employees,
            "stat_total_projects": total_projects,
            "stat_original_search_space": original_search_space,
            "stat_candidate_pairs_after_pruning": pairs_after_pruning,
            "stat_variables_created": len(prob.variables()),
            "stat_constraints_created": len(prob.constraints),
            "stat_variable_reduction_pct": round(reduction_pct, 2),
            "stat_avg_candidates_per_project": round(pairs_after_pruning / total_projects, 2) if total_projects > 0 else 0.0,
            "stat_avg_candidates_per_employee": round(pairs_after_pruning / total_employees, 2) if total_employees > 0 else 0.0
        }

        return pd.DataFrame(assignments), completed_ids

# ==========================================
# 2. MONTE CARLO SIMULATION ENGINE
# ==========================================
class LivingMonteCarloSimulator:
    """
    Stateful Simulator collecting execution traces and performance summary statistics.
    """
    def __init__(self, config: Any):
        if isinstance(config, dict):
            self.cfg = SimulationConfig.from_dict(config)
        else:
            self.cfg = config
        self.master_rng = np.random.default_rng(101)

    def _generate_workforce(self, num_emps: int) -> pd.DataFrame:
        return pd.DataFrame({
            'emp_id': [f"E{i}" for i in range(num_emps)],
            'role': self.master_rng.choice(['Backend Eng', 'Frontend Eng'], size=num_emps),
            'availability': OVERTIME_MAX_HOURS,
            'base_efficiency': 1.0,
            'efficiency': 1.0,
            'seniority': self.master_rng.uniform(0.1, 0.7, size=num_emps),
            'experience': self.master_rng.uniform(0.0, 3.0, size=num_emps),
            'rolling_fatigue': 0.0
        })

    def _generate_backlog(self, num_projs: int, start_idx: int = 0) -> pd.DataFrame:
        return pd.DataFrame({
            'project_id': [f"P{i}" for i in range(start_idx, start_idx + num_projs)],
            'required_role': self.master_rng.choice(['Backend Eng', 'Frontend Eng'], size=num_projs),
            'min_staff': self.master_rng.integers(2, 5, size=num_projs),
            'priority': self.master_rng.integers(1, 4, size=num_projs),
            'urgency': self.master_rng.integers(1, 4, size=num_projs),
            'complexity': self.master_rng.uniform(1.2, 3.0, size=num_projs),
            'deadline_days': self.master_rng.integers(15, 60, size=num_projs),
            'work_required': self.master_rng.uniform(100.0, 300.0, size=num_projs),
            'work_completed': 0.0,
            'delay_cycles': 0,
            'status': 'Pending'
        })

    def _process_workforce_evolution(self, emps: pd.DataFrame, assignments_df: pd.DataFrame, step_rng) -> pd.DataFrame:
        df = emps.copy()
        
        hours_map = {}
        if not assignments_df.empty:
            hours_map = assignments_df.groupby('emp_id')['allocated_hours'].sum().to_dict()
        df['hours_worked'] = df['emp_id'].map(hours_map).fillna(0.0)

        overtime_hours = df['hours_worked'] - STANDARD_WEEKLY_HOURS
        fatigue_deltas = np.where(overtime_hours > 0, overtime_hours / STANDARD_WEEKLY_HOURS, -0.25)
        df['rolling_fatigue'] = (df['rolling_fatigue'] + fatigue_deltas).clip(0.0, 2.0)
        
        df['efficiency'] = df['base_efficiency'] * (1.0 - 0.15 * df['rolling_fatigue'])

        df['experience'] += np.where(df['hours_worked'] > 0, self.cfg.experience_growth_rate, 0.0)
        promoted = df['experience'] >= self.cfg.promotion_threshold
        df.loc[promoted, 'base_efficiency'] *= 1.10
        df.loc[promoted, 'experience'] = 0.0
        df.loc[promoted, 'seniority'] += 0.10

        retire_prob = np.where(df['seniority'] > self.cfg.retirement_threshold, self.cfg.retirement_rate, 0.0)
        retired = step_rng.random(len(df)) < retire_prob
        df = df[~retired].reset_index(drop=True)

        return df

    def _intelligent_hiring(self, emps: pd.DataFrame, backlog_df: pd.DataFrame, trial: int, step: int) -> pd.DataFrame:
        if backlog_df.empty:
            return emps

        role_demands = backlog_df.groupby('required_role')['min_staff'].sum().to_dict()
        role_supply = emps.groupby('role')['emp_id'].count().to_dict()
        
        gap_role = 'Backend Eng'
        max_gap = -999
        for r in ['Backend Eng', 'Frontend Eng']:
            gap = role_demands.get(r, 0) - role_supply.get(r, 0)
            if gap > max_gap:
                max_gap = gap
                gap_role = r

        new_hires = pd.DataFrame({
            'emp_id': [f"HIRE_{trial}_{step}_{i}" for i in range(self.cfg.hires_per_batch)],
            'role': [gap_role] * self.cfg.hires_per_batch,
            'availability': OVERTIME_MAX_HOURS,
            'base_efficiency': 0.8,
            'efficiency': 0.8,
            'seniority': 0.1,
            'experience': 0.0,
            'rolling_fatigue': 0.0
        })
        return pd.concat([emps, new_hires], ignore_index=True)

    def run_simulation(self) -> Dict[str, Any]:
        t_sim_start = time.perf_counter()
        
        all_allocations = []
        all_kpis = []
        all_burnouts = []
        project_history = []
        optimizer_telemetry = []
        
        init_emps = self._generate_workforce(self.cfg.initial_employees)
        init_backlog = self._generate_backlog(self.cfg.initial_projects)

        global_proj_idx = self.cfg.initial_projects

        for trial in range(self.cfg.trials):
            trial_seed = int(self.master_rng.integers(1, 1_000_000))
            trial_rng = np.random.default_rng(trial_seed)

            trial_emps = init_emps.copy()
            trial_backlog = init_backlog.copy()

            for step in range(self.cfg.steps_per_trial):
                event = trial_rng.choice(['Normal Operations', 'Systemic Blackout', 'Crunch Culture Spike'], p=[0.75, 0.15, 0.10])
                absence_rate = trial_rng.beta(self.cfg.absence_alpha, self.cfg.absence_beta)
                absent_mask = trial_rng.random(len(trial_emps)) < absence_rate
                
                active_emps = trial_emps.copy()
                active_emps.loc[absent_mask, 'availability'] = 0.0

                if event == 'Systemic Blackout':
                    active_emps['availability'] = (active_emps['availability'] * 0.75).round()
                elif event == 'Crunch Culture Spike':
                    trial_backlog['urgency'] = np.minimum(trial_backlog['urgency'] + 1, 3)

                optimizer = WorkforceOptimizer(
                    active_emps, 
                    trial_backlog[trial_backlog['status'] != 'Completed'],
                    top_k_multiplier=self.cfg.top_k_multiplier
                )
                df_assign, completed_ids = optimizer.run_allocation(trial_seed=trial_seed + step)

                if optimizer.metrics:
                    step_metrics = optimizer.metrics.copy()
                    step_metrics['trial'] = trial + 1
                    step_metrics['step'] = step + 1
                    optimizer_telemetry.append(step_metrics)

                if not df_assign.empty:
                    work_done = df_assign.groupby('project_id')['effective_output_hours'].sum().to_dict()
                    trial_backlog['work_completed'] += trial_backlog['project_id'].map(work_done).fillna(0.0)

                    assigned_proj_ids = df_assign['project_id'].unique()
                    delayed_mask = (~trial_backlog['project_id'].isin(assigned_proj_ids)) & (trial_backlog['status'] != 'Completed')
                    trial_backlog.loc[delayed_mask, 'delay_cycles'] += 1

                completed_mask = (trial_backlog['work_completed'] >= trial_backlog['work_required']) & (trial_backlog['status'] != 'Completed')
                trial_backlog.loc[completed_mask, 'status'] = 'Completed'

                trial_backlog['deadline_days'] -= 5
                failed_mask = (trial_backlog['deadline_days'] <= 0) & (trial_backlog['status'] != 'Completed')
                trial_backlog.loc[failed_mask, 'status'] = 'Failed'

                t_post_start = time.perf_counter()
                trial_emps = self._process_workforce_evolution(trial_emps, df_assign, trial_rng)

                if step % self.cfg.hiring_interval_steps == 0:
                    trial_emps = self._intelligent_hiring(trial_emps, trial_backlog[trial_backlog['status'] == 'Pending'], trial, step)

                new_projs = self._generate_backlog(self.cfg.new_projects_per_step, start_idx=global_proj_idx)
                global_proj_idx += self.cfg.new_projects_per_step
                trial_backlog = pd.concat([trial_backlog, new_projs], ignore_index=True)

                total_capacity = active_emps['availability'].sum()
                total_allocated = df_assign['allocated_hours'].sum() if not df_assign.empty else 0.0
                utilization = (total_allocated / total_capacity * 100.0) if total_capacity > 0 else 0.0

                role_match_rate = df_assign['role_match'].mean() if not df_assign.empty else 0.0
                avg_delay = df_assign['delay_factor'].mean() if not df_assign.empty else 1.0

                active_count = len(active_emps)
                contention_index = len(trial_backlog[trial_backlog['status'] == 'Pending']) / active_count if active_count > 0 else 0.0
                
                if optimizer_telemetry:
                    optimizer_telemetry[-1]['time_simulation_post_processing'] = time.perf_counter() - t_post_start

                all_kpis.append({
                    'Trial': trial + 1,
                    'Step': step + 1,
                    'Event': event,
                    'Utilization (%)': round(utilization, 2),
                    'Projects Completed': len(trial_backlog[trial_backlog['status'] == 'Completed']),
                    'Projects Failed/Shelved': len(trial_backlog[trial_backlog['status'] == 'Failed']),
                    'Active Backlog Size': len(trial_backlog[trial_backlog['status'] == 'Pending']),
                    'Avg Burnout': round(trial_emps['rolling_fatigue'].mean(), 3),
                    'Role Match Rate': round(role_match_rate, 2),
                    'Avg Project Delay (Factor)': round(avg_delay, 2),
                    'Resource Contention': round(contention_index, 3)
                })

                if not df_assign.empty:
                    df_assign['trial'] = trial + 1
                    df_assign['step'] = step + 1
                    all_allocations.append(df_assign)

                trial_emps_burnout = trial_emps[['emp_id', 'rolling_fatigue']].copy()
                trial_emps_burnout['trial'] = trial + 1
                trial_emps_burnout['step'] = step + 1
                all_burnouts.append(trial_emps_burnout)

            trial_backlog['trial'] = trial + 1
            project_history.append(trial_backlog)

        t_sim_end = time.perf_counter()
        t_total_simulation = t_sim_end - t_sim_start

        allocations_final_df = pd.concat(all_allocations, ignore_index=True) if all_allocations else pd.DataFrame()
        kpis_final_df = pd.DataFrame(all_kpis)
        burnout_final_df = pd.concat(all_burnouts, ignore_index=True) if all_burnouts else pd.DataFrame()
        projects_final_df = pd.concat(project_history, ignore_index=True) if project_history else pd.DataFrame()
        df_telemetry = pd.DataFrame(optimizer_telemetry)

        performance_summary = {}
        if not df_telemetry.empty:
            total_opt_time = df_telemetry["time_total_optimizer"].sum()
            total_solver_time = df_telemetry["time_solver_execution"].sum()
            total_post_time = df_telemetry.get("time_simulation_post_processing", pd.Series([0.0])).sum() + df_telemetry["time_solution_extraction"].sum()
            
            total_build_time = (
                df_telemetry["time_model_build"].sum() + 
                df_telemetry["time_candidate_gen_prune"].sum() + 
                df_telemetry["time_adjacency_build"].sum()
            )
            
            solver_pct = (total_solver_time / t_total_simulation * 100.0) if t_total_simulation > 0 else 0.0
            build_pct = (total_build_time / t_total_simulation * 100.0) if t_total_simulation > 0 else 0.0
            
            performance_summary = {
                "total_simulation_time": round(t_total_simulation, 4),
                "total_optimization_time": round(total_opt_time, 4),
                "total_model_build_time": round(total_build_time, 4),
                "total_solver_time": round(total_solver_time, 4),
                "total_post_processing_time": round(total_post_time, 4),
                "solver_percentage_of_total_runtime": round(solver_pct, 2),
                "python_build_percentage_of_total_runtime": round(build_pct, 2),
                "avg_solve_time_per_cycle": round(df_telemetry["time_solver_execution"].mean(), 4),
                "avg_variables_created": round(df_telemetry["stat_variables_created"].mean(), 1),
                "avg_constraints_created": round(df_telemetry["stat_constraints_created"].mean(), 1),
                "avg_search_space_reduction_pct": round(df_telemetry["stat_variable_reduction_pct"].mean(), 2),
                "avg_solve_time_per_trial": round(df_telemetry.groupby("trial")["time_solver_execution"].sum().mean(), 4)
            }

        dept_summary_df = pd.DataFrame()
        if not allocations_final_df.empty:
            dept_summary_df = allocations_final_df.groupby(['trial', 'role']).agg(
                total_hours_allocated=('allocated_hours', 'sum'),
                avg_output_delivered=('effective_output_hours', 'mean'),
                role_match_rate=('role_match', 'mean')
            ).reset_index()

        project_summary_df = projects_final_df.groupby('status').size().reset_index(name='count')
        simulation_summary_df = kpis_final_df.describe()

        return {
            "allocations": json.loads(allocations_final_df.to_json(orient='records')),
            "employees": json.loads(trial_emps.to_json(orient='records')),
            "projects": json.loads(projects_final_df.to_json(orient='records')),
            "kpis": json.loads(kpis_final_df.to_json(orient='records')),
            "burnout": json.loads(burnout_final_df.to_json(orient='records')),
            "departments": json.loads(dept_summary_df.to_json(orient='records')) if not dept_summary_df.empty else [],
            "project_summary": json.loads(project_summary_df.to_json(orient='records')),
            "simulation_summary": json.loads(simulation_summary_df.to_json(orient='index')),
            
            "performance_summary": performance_summary,
            "performance_details": json.loads(df_telemetry.to_json(orient='records')) if not df_telemetry.empty else []
        }

# ==========================================
# 3. INTERACTIVE CLI ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine.py <path_to_config.json> [path_to_output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results.json"

    try:
        with open(input_path, 'r') as f:
            config_payload = json.load(f)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)

    print(f"Initializing simulation using config from: {input_path}")
    start_time = time.time()
    
    simulator = LivingMonteCarloSimulator(config=config_payload)
    results = simulator.run_simulation()
    
    elapsed_time = time.time() - start_time
    print(f"Simulation completed successfully in {elapsed_time:.2f} seconds.")

    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results exported to: {output_path}")
    except Exception as e:
        print(f"Error exporting results to JSON: {e}")
        sys.exit(1)
