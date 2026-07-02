# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.rcParams["axes3d.mouserotationstyle"] = "azel"
plt.rcParams.update({
    'font.family': 'Century Schoolbook',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
    'mathtext.it': 'serif:italic',
    'mathtext.bf': 'serif:bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'legend.title_fontsize': 10
})

COLORS = {"GUROBI": "#EC3826", "SCIP": "#2E8B57", "HIGHS": "#0055A4"}
MARKERS = {"Triangle": "^", "Square": "s", "DC": "h", "Quadratic": "o", "GUROBI": "o", "SCIP": "o", "HIGHS": "o"}

def get_color(solver_name):
    return COLORS.get(str(solver_name).upper(), "#333333")

def load_data(filepath):
    df = pd.read_csv(filepath, index_col=list(range(8))).reset_index()
    df['Is_Optimal'] = (df['Status'] == 'OPTIMAL')
    df['Solver'] = df['Solver'].astype(str)
    df['Problem type'] = df['Problem type'].astype(str).str.upper()
    df['CPWL representation'] = df['CPWL representation'].fillna("Quadratic").astype(str)
    df['Degree of accuracy'] = pd.to_numeric(df['Degree of accuracy'], errors='coerce').fillna(0).astype(int)
    return df

def plot_faces_3d(faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poly_collection = Poly3DCollection(faces, facecolors='C1', linewidths=1, edgecolors='black', alpha=0.6)
    ax.add_collection3d(poly_collection)
    return

def plot_efficiency_grid(df, statistic="mean", time_limit=600, y_max=650, output_dir="."):
    df_ncqp = df[df["Instance family"] == "NCQP"].copy()
    sequence_lengths = [24, 48, 96, 168]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150, sharey=True)
    axes = axes.ravel()
    legend_handles = {}

    for i, (ax, T) in enumerate(zip(axes, sequence_lengths)):
        sub_df = df_ncqp[df_ncqp["Sequence length"] == T]
        agg_df = sub_df.groupby(["Problem type", "Solver", "CPWL representation", "Degree of accuracy"], as_index=False).agg(solve_time=("Solve time", statistic))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        df_milp = agg_df[agg_df["Problem type"] == "MILP"]
        for (solver, method), sub in df_milp.groupby(["Solver", "CPWL representation"]):
            sub = sub.sort_values("Degree of accuracy")
            label = f"{solver}, {method}"
            line, = ax.plot(sub["Degree of accuracy"], sub["solve_time"], color=get_color(solver), marker=MARKERS.get(method, "o"), markersize=8, markerfacecolor="none", label=label)
            legend_handles[label] = line

        df_qp = agg_df[agg_df["Problem type"].isin(["QP", "QCP", "MIQP", "MIQCP"])]
        for solver, sub in df_qp.groupby("Solver"):
            label = f"{solver}, Quadratic"
            line = ax.axhline(y=sub["solve_time"].mean(), color=get_color(solver), linestyle="--", label=label)
            legend_handles[label] = line

        ax.set_ylim([0, y_max])
        current_yticks = [y for y in ax.get_yticks() if 0 <= y <= y_max]
        new_yticks = sorted(set(current_yticks + [time_limit]))
        ax.set_yticks(new_yticks)
        ax.set_yticklabels(["Time limit" if np.isclose(y, time_limit) else f"{y:g}" for y in new_yticks])
        ax.set_xticks(sorted(df_milp["Degree of accuracy"].unique()))
        ax.set_title(fr"$T = {T}$")
        ax.grid(True, alpha=0.25)
        if i >= 2: ax.set_xlabel(r"Degree of accuracy $n$", labelpad=6)
        if i in [0, 2]: ax.set_ylabel(f"{statistic.capitalize()} solve time (s)", labelpad=-16)

    fig.legend(handles=list(legend_handles.values()), labels=list(legend_handles.keys()), loc="lower center", bbox_to_anchor=(0.5, 0.00), ncol=4, frameon=False)
    fig.tight_layout(rect=[0.02, 0.12, 1.00, 1.00])
    
    filepath = os.path.join(output_dir, "NCQP_PWL_MILP_efficiency_2x2.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_scalability(df, output_dir="."):
    df_ncqp = df[df['Instance family'] == 'NCQP'].copy()
    seq_lengths = sorted(df_ncqp['Sequence length'].dropna().unique())
    x = np.array(seq_lengths)
    
    configs = [('QP', 'SCIP', 'QP (SCIP)'), ('QP', 'Gurobi', 'QP (Gurobi)'), ('MILP', 'HiGHS', r'MILP (HiGHS)')]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)

    for prob_type, solver, label in configs:
        rates = []
        for seq in seq_lengths:
            mask = (df_ncqp['Sequence length'] == seq) & (df_ncqp['Problem type'] == prob_type) & (df_ncqp['Solver'] == solver)
            if prob_type == 'MILP': mask &= (df_ncqp['CPWL representation'] == 'Square') & (df_ncqp['Degree of accuracy'] == 4)
            subset = df_ncqp[mask]
            rates.append(subset['Is_Optimal'].mean() * 100 if not subset.empty else 0)
            
        ax.plot(x, rates, label=label, color=get_color(solver), marker=MARKERS.get(solver.upper(), "o"), linewidth=2, markersize=8)

    ax.set_ylabel('Solve rate')
    ax.set_xlabel('Sequence length (T)')
    ax.set_xticks(x)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "Scalability.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_dolan_more(df, output_dir="."):
    df_perf = df[(df['Instance family'] == 'NCQP') & (df['Solver'] == 'HiGHS')].dropna(subset=['Solve time']).copy()
    time_pivot = df_perf.pivot_table(index='Instance', columns='CPWL representation', values='Solve time')
    ratios = time_pivot.divide(time_pivot.min(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    colors = {'Square': 'C0', 'Triangle': 'C1', 'DC': 'C2'}
    
    for col in ratios.columns:
        sorted_ratios = np.sort(ratios[col].dropna())
        y_vals = (np.arange(1, len(sorted_ratios) + 1) / len(time_pivot)) * 100
        ax.step(sorted_ratios, y_vals, label=col, where='post', linewidth=2, color=colors.get(col))

    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'Performance Ratio $\tau$ (log$_2$ scale)')
    ax.set_ylabel('Percentage of instances')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
    ax.set_ylim(-5, 105)
    ax.legend(title='Formulation', loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "DolanMore_rep.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_applicability(df, output_dir="."):
    mask_qp = df['Problem type'] == 'QP'
    mask_milp = (df['Problem type'] == 'MILP') & (df['CPWL representation'] == 'Square') & (df['Degree of accuracy'] == 4)
    df_filtered = df[mask_qp | mask_milp].copy()
    agg = df_filtered.groupby(['Instance family', 'Problem type', 'Solver'])['Is_Optimal'].mean().reset_index()
    
    families = ['NCQP', 'QPLIB']
    methods = [('QP', 'Gurobi'), ('QP', 'SCIP'), ('MILP', 'HiGHS')]
    x = np.arange(len(families))
    width = 0.2
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    
    for i, (prob_type, solver) in enumerate(methods):
        rates = [(agg[(agg['Instance family'] == fam) & (agg['Problem type'] == prob_type) & (agg['Solver'] == solver)]['Is_Optimal'].values[0] * 100) if not agg[(agg['Instance family'] == fam) & (agg['Problem type'] == prob_type) & (agg['Solver'] == solver)]['Is_Optimal'].empty else 0 for fam in families]
        ax.bar(x + (width * i), rates, width, label=f'{prob_type} ({solver})', color=get_color(solver), edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Solve rate')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['SCBP', 'QPLIB'])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(100.0))
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "Applicability.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def plot_approx_gap(df, output_dir="."):
    df_gap = df[(df["Instance family"] == "NCQP") & (df["Problem type"] == "MILP") & (df["Solver"] == "HiGHS") & (df["CPWL representation"] == "Square") & (df["Degree of accuracy"].isin([1, 2, 3, 4]))].copy()
    df_gap["Objective CPWL approximation gap (%)"] = ((df_gap["Objective value"] - df_gap["Quadratic objective value"]) / df_gap["Quadratic objective value"]) * 100
    
    degrees = [1, 2, 3, 4]
    box_data = [df_gap.loc[df_gap["Degree of accuracy"] == n, "Objective CPWL approximation gap (%)"].dropna() for n in degrees]
    
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.boxplot(box_data, positions=degrees, widths=0.3, showfliers=True)
    ax.plot(degrees, [np.median(arr) for arr in box_data], 'k', linestyle='--', linewidth=1.0)
    
    ax.set_xticks(degrees)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_xlabel(r"Degree of accuracy $n$")
    ax.set_ylabel("Objective CPWL approximation gap (%)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    
    filepath = os.path.join(output_dir, "Obj_CPWL_approx_gap.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

def generate_qplib_latex_table(df):
    # (Remains unchanged as it prints to console)
    df_qplib = df[df['Instance family'] == 'QPLIB'].copy()
    df_qp = df_qplib[(df_qplib['Problem type'] == 'QP') & (df_qplib['Solver'] == 'SCIP')][['Instance', 'Objective value', 'Dual bound', 'Relative gap']].rename(columns={'Objective value': 'Primal_QP', 'Dual bound': 'Dual_QP', 'Relative gap': 'Gap_QP'})
    df_milp = df_qplib[(df_qplib['Problem type'] == 'MILP') & (df_qplib['Solver'] == 'HiGHS') & (df_qplib['CPWL representation'] == 'Square') & (df_qplib['Degree of accuracy'] == 3)][['Instance', 'Objective value', 'Dual bound', 'Relative gap']].rename(columns={'Objective value': 'Primal_MILP', 'Dual bound': 'Dual_MILP', 'Relative gap': 'Gap_MILP'})
    
    comp_df = pd.merge(df_qp, df_milp, on='Instance', how='inner')
    comp_df['Gap_QP'] *= 100
    comp_df['Gap_MILP'] *= 100
    
    df_latex = comp_df.set_index("Instance").sort_index()
    df_latex.columns = pd.MultiIndex.from_tuples([("QP (SCIP)", "Primal"), ("QP (SCIP)", "Dual"), ("QP (SCIP)", r"Gap (\%)"), ("MILP (HiGHS)", "Primal"), ("MILP (HiGHS)", "Dual"), ("MILP (HiGHS)", r"Gap (\%)")])
    
    # latex_table = df_latex.to_latex(index=True, multicolumn=True, multicolumn_format="c", escape=False, float_format="%.2f")
    # print("=== QPLIB LaTeX Table ===")
    # print(latex_table)
    print(df_latex)
    return df_latex