import time
import numpy as np
import pandas as pd
import pulp
from dataclasses import dataclass, field
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

# ==========================================
# 1. OPTIMIZATION ENGINE (MILP IMPLEMENTATION)
# ==========================================
class WorkforceOptimizer:
    """
    MILP Optimizer utilizing PuLP to allocate workers to a persistent backlog queue.
    """
    def __init__(self, employees_df: pd.DataFrame, backlog_df: pd.DataFrame):
        self.employees_df = employees_df.copy()
        self.backlog_df = backlog_df.copy()

    def run_allocation(self, trial_seed: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        if self.employees_df.empty or self.backlog_df.empty:
            return pd.DataFrame(), []

        # Setup random generators
        rng = np.random.default_rng(trial_seed)
        
        # Prepare lookup objects for fast vectorized iteration
        df_proj = self.backlog_df.copy()
        df_proj['score'] = df_proj['priority'] * df_proj['urgency']
        df_proj = df_proj.sort_values(by='score', ascending=False)

        emp_ids = self.employees_df['emp_id'].tolist()
        proj_ids = df_proj['project_id'].tolist()

        # Extract worker parameters for lookup
        emp_roles = self.employees_df.set_index('emp_id')['role'].to_dict()
        emp_avails = self.employees_df.set_index('emp_id')['availability'].to_dict()
        emp_effs = self.employees_df.set_index('emp_id')['efficiency'].to_dict()
        emp_fatigue = self.employees_df.set_index('emp_id')['rolling_fatigue'].to_dict()

        # Extract project parameters for lookup
        proj_roles = df_proj.set_index('project_id')['required_role'].to_dict()
        proj_min_staff = df_proj.set_index('project_id')['min_staff'].to_dict()
        proj_complexity = df_proj.set_index('project_id')['complexity'].to_dict()
        proj_scores = df_proj.set_index('project_id')['score'].to_dict()
        proj_priority = df_proj.set_index('project_id')['priority'].to_dict() # FIX: Dict mapping for loop performance

        # 1. Decision Variables
        prob = pulp.LpProblem("Workforce_Allocation_Optimization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("assign", ((i, j) for i in emp_ids for j in proj_ids), cat=pulp.LpBinary)
        shortfall = pulp.LpVariable.dicts("shortfall", proj_ids, lowBound=0, cat=pulp.LpInteger)

        # 2. Compute Utility & Penalty Matrices
        utility = {}
        for i in emp_ids:
            for j in proj_ids:
                is_match = (emp_roles[i] == proj_roles[j])
                role_modifier = 1.0 if is_match else CROSS_ROLE_PENALTY
                fatigue_modifier = 1.0 - (0.1 * emp_fatigue[i])
                
                # Base Utility calculation
                utility[i, j] = proj_scores[j] * emp_effs[i] * role_modifier * fatigue_modifier

        # 3. Objective Function Formulation
        # Maximize allocation utility minus shortfall penalty
        prob += (
            pulp.lpSum(utility[i, j] * x[i, j] for i in emp_ids for j in proj_ids) - 
            pulp.lpSum(shortfall[j] * (50.0 * proj_scores[j]) for j in proj_ids)
        )

        # 4. Constraints Formulation
        # A. Worker hour limits
        for i in emp_ids:
            prob += pulp.lpSum(
                (BASE_HOURS_PER_PROJECT + int(10 * (proj_complexity[j] - 1.0))) * x[i, j] 
                for j in proj_ids
            ) <= emp_avails[i]

        # B. Concurrency limit (Max 2 projects per step)
        for i in emp_ids:
            prob += pulp.lpSum(x[i, j] for j in proj_ids) <= MAX_CONCURRENT_PROJECTS

        # C. Staffing Bounds (Brooks' Law Ceiling & Slack Shortfalls)
        for j in proj_ids:
            prob += pulp.lpSum(x[i, j] for i in emp_ids) + shortfall[j] >= proj_min_staff[j]
            prob += pulp.lpSum(x[i, j] for i in emp_ids) <= (proj_min_staff[j] + 2)

        # Solve MILP
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=5)
        status = prob.solve(solver)

        assignments = []
        completed_ids = []

        if pulp.LpStatus[status] == "Optimal":
            # Collect results natively
            for j in proj_ids:
                allocated_workers = [i for i in emp_ids if x[i, j].varValue == 1]
                staff_count = len(allocated_workers)
                
                if staff_count > 0:
                    needed = proj_min_staff[j]
                    delay_factor = (needed / staff_count) if staff_count > 0 else 2.0
                    eff_penalty = PARTIAL_STAFF_PENALTY if staff_count < needed else 1.0
                    
                    # Accumulate project deliverables
                    completed_ids.append(j)
                    
                    for i in allocated_workers:
                        allocated_h = BASE_HOURS_PER_PROJECT + int(10 * (proj_complexity[j] - 1.0))
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
                            'priority': proj_priority[j],  # FIX: Replaced slow runtime .loc with O(1) Dictionary Lookup
                            'role': emp_roles[i]  # FIX: Directly track role to preserve hired employees in metrics
                        })

        return pd.DataFrame(assignments), completed_ids

# ==========================================
# 2. MONTE CARLO SIMULATION ENGINE
# ==========================================
class LivingMonteCarloSimulator:
    """
    Stateful Simulator containing the persistent workforce and project backlogs.
    """
    def __init__(self, config: SimulationConfig):
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
        # FIX: Replaced legacy '.randint' calls with new Generator-compatible '.integers'
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
        
        # 1. Calculate and Accumulate Hours Worked
        hours_map = {}
        if not assignments_df.empty:
            hours_map = assignments_df.groupby('emp_id')['allocated_hours'].sum().to_dict()
        df['hours_worked'] = df['emp_id'].map(hours_map).fillna(0.0)

        # 2. Burnout Accumulation and Recovery Formulas (Standardized)
        overtime_hours = df['hours_worked'] - STANDARD_WEEKLY_HOURS
        fatigue_deltas = np.where(overtime_hours > 0, overtime_hours / STANDARD_WEEKLY_HOURS, -0.25)
        df['rolling_fatigue'] = (df['rolling_fatigue'] + fatigue_deltas).clip(0.0, 2.0)
        
        # Adjust productivity/efficiency dynamically
        df['efficiency'] = df['base_efficiency'] * (1.0 - 0.15 * df['rolling_fatigue'])

        # 3. Career Path Evolution (Experience and Promotion)
        df['experience'] += np.where(df['hours_worked'] > 0, self.cfg.experience_growth_rate, 0.0)
        promoted = df['experience'] >= self.cfg.promotion_threshold
        df.loc[promoted, 'base_efficiency'] *= 1.10
        df.loc[promoted, 'experience'] = 0.0
        df.loc[promoted, 'seniority'] += 0.10

        # 4. Senior Retirements
        retire_prob = np.where(df['seniority'] > self.cfg.retirement_threshold, self.cfg.retirement_rate, 0.0)
        retired = step_rng.random(len(df)) < retire_prob
        df = df[~retired].reset_index(drop=True)

        return df

    def _intelligent_hiring(self, emps: pd.DataFrame, backlog_df: pd.DataFrame, trial: int, step: int) -> pd.DataFrame:
        """
        Calculates role gaps based on the active backlog to intelligently target talent acquisition.
        """
        if backlog_df.empty:
            return emps

        role_demands = backlog_df.groupby('required_role')['min_staff'].sum().to_dict()
        role_supply = emps.groupby('role')['emp_id'].count().to_dict()
        
        # Compute bottleneck role
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
            'base_efficiency': 0.8,  # Juniors start slightly lower
            'efficiency': 0.8,
            'seniority': 0.1,
            'experience': 0.0,
            'rolling_fatigue': 0.0
        })
        return pd.concat([emps, new_hires], ignore_index=True)

    def run_simulation(self) -> Dict[str, pd.DataFrame]:
        """
        Main runner executing the Monte Carlo loops across persistent steps.
        """
        all_allocations = []
        all_kpis = []
        all_burnouts = []
        project_history = []
        
        # Initialize master data pools
        init_emps = self._generate_workforce(self.cfg.initial_employees)
        init_backlog = self._generate_backlog(self.cfg.initial_projects)

        global_proj_idx = self.cfg.initial_projects

        for trial in range(self.cfg.trials):
            trial_seed = int(self.master_rng.integers(1, 1_000_000))
            trial_rng = np.random.default_rng(trial_seed)

            # Deep copy to maintain state independence per trial
            trial_emps = init_emps.copy()
            trial_backlog = init_backlog.copy()

            for step in range(self.cfg.steps_per_trial):
                # 1. Environmental Chaos and Random Events
                event = trial_rng.choice(['Normal Operations', 'Systemic Blackout', 'Crunch Culture Spike'], p=[0.75, 0.15, 0.10])
                
                # Randomized worker absences using Beta distribution
                absence_rate = trial_rng.beta(self.cfg.absence_alpha, self.cfg.absence_beta)
                absent_mask = trial_rng.random(len(trial_emps)) < absence_rate
                
                active_emps = trial_emps.copy()
                active_emps.loc[absent_mask, 'availability'] = 0.0

                # Environmental Modifier Impacts
                if event == 'Systemic Blackout':
                    active_emps['availability'] = (active_emps['availability'] * 0.75).round()
                elif event == 'Crunch Culture Spike':
                    trial_backlog['urgency'] = np.minimum(trial_backlog['urgency'] + 1, 3)

                # 2. Run Intelligent Allocation
                optimizer = WorkforceOptimizer(active_emps, trial_backlog[trial_backlog['status'] != 'Completed'])
                df_assign, completed_ids = optimizer.run_allocation(trial_seed=trial_seed + step)

                # 3. Calculate Project Progression / Backlog updates
                if not df_assign.empty:
                    work_done = df_assign.groupby('project_id')['effective_output_hours'].sum().to_dict()
                    trial_backlog['work_completed'] += trial_backlog['project_id'].map(work_done).fillna(0.0)

                    # Mark Delayed items
                    assigned_proj_ids = df_assign['project_id'].unique()
                    delayed_mask = (~trial_backlog['project_id'].isin(assigned_proj_ids)) & (trial_backlog['status'] != 'Completed')
                    trial_backlog.loc[delayed_mask, 'delay_cycles'] += 1

                # Update lifecycle status
                completed_mask = (trial_backlog['work_completed'] >= trial_backlog['work_required']) & (trial_backlog['status'] != 'Completed')
                trial_backlog.loc[completed_mask, 'status'] = 'Completed'

                # Progress time constraints
                trial_backlog['deadline_days'] -= 5
                failed_mask = (trial_backlog['deadline_days'] <= 0) & (trial_backlog['status'] != 'Completed')
                trial_backlog.loc[failed_mask, 'status'] = 'Failed'

                # 4. Apply System Dynamics
                trial_emps = self._process_workforce_evolution(trial_emps, df_assign, trial_rng)

                if step % self.cfg.hiring_interval_steps == 0:
                    trial_emps = self._intelligent_hiring(trial_emps, trial_backlog[trial_backlog['status'] == 'Pending'], trial, step)

                # Generate dynamic scope creep / organic growth
                new_projs = self._generate_backlog(self.cfg.new_projects_per_step, start_idx=global_proj_idx)
                global_proj_idx += self.cfg.new_projects_per_step
                trial_backlog = pd.concat([trial_backlog, new_projs], ignore_index=True)

                # 5. Metric Calculations
                total_capacity = active_emps['availability'].sum()
                total_allocated = df_assign['allocated_hours'].sum() if not df_assign.empty else 0.0
                utilization = (total_allocated / total_capacity * 100.0) if total_capacity > 0 else 0.0

                role_match_rate = df_assign['role_match'].mean() if not df_assign.empty else 0.0
                avg_delay = df_assign['delay_factor'].mean() if not df_assign.empty else 1.0

                # Resource Contention: Projects needing staff vs. available workforce capacity
                # FIX: Added a defensive Division-by-Zero check in case the workforce drops to 0
                active_count = len(active_emps)
                contention_index = len(trial_backlog[trial_backlog['status'] == 'Pending']) / active_count if active_count > 0 else 0.0

                # Append Logging State
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

            # Record final statuses for the trial
            trial_backlog['trial'] = trial + 1
            project_history.append(trial_backlog)

        # ==========================================
        # MODULE: METRICS, ANALYTICS & EXPORTS
        # ==========================================
        allocations_final_df = pd.concat(all_allocations, ignore_index=True) if all_allocations else pd.DataFrame()
        kpis_final_df = pd.DataFrame(all_kpis)
        burnout_final_df = pd.concat(all_burnouts, ignore_index=True) if all_burnouts else pd.DataFrame()
        projects_final_df = pd.concat(project_history, ignore_index=True) if project_history else pd.DataFrame()

        # Build Department-level summary aggregated metrics
        dept_summary_df = pd.DataFrame()
        if not allocations_final_df.empty:
            # FIX: By tracking 'role' during allocation, we bypass the need for a left-merge that drops hired employees
            dept_summary_df = allocations_final_df.groupby(['trial', 'role']).agg(
                total_hours_allocated=('allocated_hours', 'sum'),
                avg_output_delivered=('effective_output_hours', 'mean'),
                role_match_rate=('role_match', 'mean')
            ).reset_index()

        # Compile final consolidated outputs
        return {
            "allocations": allocations_final_df,
            "employees": trial_emps,  # Representative slice of the last updated workforce state
            "projects": projects_final_df,
            "kpis": kpis_final_df,
            "burnout": burnout_final_df,
            "departments": dept_summary_df,
            "project_summary": projects_final_df.groupby('status').size().reset_index(name='count'),
            "simulation_summary": kpis_final_df.describe()
        }
