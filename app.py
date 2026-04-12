"""
TSP Optimizer — Hill Climbing vs Simulated Annealing
=====================================================
Interactive Streamlit application comparing two optimization algorithms
for the Traveling Salesman Problem: Hill Climbing (local search) and
Simulated Annealing (metaheuristic with probabilistic exploration).

Usage:
    streamlit run app.py

Features:
    - Load TSP instances from Kaggle TSPLIB dataset or generate random cities
    - Side-by-side route visualization with Plotly
    - Convergence analysis with overlaid distance curves
    - Temperature decay visualization for Simulated Annealing
    - Batch experiment mode: run both algorithms on multiple instances
    - Interactive parameter tuning via sidebar controls
"""

import streamlit as st
import numpy as np
import pandas as pd
import time

from src.algorithms.hill_climbing import hill_climbing
from src.algorithms.simulated_annealing import simulated_annealing
from src.models.tsp import (
    generate_initial_route,
    generate_random_cities,
    calculate_total_distance,
)
from src.utils.data_loader import (
    load_dataset,
    parse_instance,
    get_instance_ids,
    detect_n_cities,
    get_instance_by_id,
)
from src.utils.visualization import (
    plot_route,
    plot_convergence,
    plot_temperature_decay,
    plot_comparison_bar,
    plot_batch_results,
    plot_batch_scatter,
    HC_COLOR,
    SA_COLOR,
)

# ══════════════════════════════════════════════════════════════════════
#  Page Configuration
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TSP Optimizer — HC vs SA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
#  Custom CSS — Premium Dark Theme
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --hc-color: #FF6B6B;
        --sa-color: #4ECDC4;
        --bg-dark: #0E1117;
        --card-bg: rgba(255,255,255,0.04);
        --card-border: rgba(255,255,255,0.08);
        --text-primary: #E0E0E0;
        --text-secondary: #888;
        --accent-gold: #FFD93D;
    }

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        background: linear-gradient(135deg, rgba(255,107,107,0.1) 0%, rgba(78,205,196,0.1) 100%);
        border-radius: 16px;
        border: 1px solid var(--card-border);
        margin-bottom: 2rem;
    }

    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .hero-header p {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, border-color 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,0.15);
    }

    .metric-card .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-bottom: 0.3rem;
    }

    .metric-card .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
    }

    .metric-card .metric-sublabel {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.2rem;
    }

    /* Winner badge */
    .winner-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        margin: 0.5rem 0;
    }

    .winner-hc {
        background: rgba(255,107,107,0.15);
        border: 1px solid rgba(255,107,107,0.4);
        color: #FF6B6B;
    }

    .winner-sa {
        background: rgba(78,205,196,0.15);
        border: 1px solid rgba(78,205,196,0.4);
        color: #4ECDC4;
    }

    .winner-tie {
        background: rgba(255,217,61,0.15);
        border: 1px solid rgba(255,217,61,0.4);
        color: #FFD93D;
    }

    /* Algorithm tags */
    .algo-tag-hc {
        color: #FF6B6B;
        font-weight: 600;
    }

    .algo-tag-sa {
        color: #4ECDC4;
        font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1a1a2e 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        margin: 1.5rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(78,205,196,0.08);
        border-left: 3px solid #4ECDC4;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <h1> TSP Optimizer</h1>
    <p>Comparing Hill Climbing vs Simulated Annealing for the Traveling Salesman Problem</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Sidebar Controls
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("##  Configuration")
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Data Source Selection ──
    st.markdown("###  Data Source")
    data_mode = st.radio(
        "Choose data source:",
        [" Kaggle / Sample Dataset", " Random Cities"],
        help="Primary: Kaggle TSPLIB dataset. Fallback: bundled sample CSV."
    )

    dataset_loaded = False
    df = None
    cities = None
    n_cities = 20

    if data_mode == " Kaggle / Sample Dataset":
        with st.spinner("Loading dataset..."):
            df, source_name = load_dataset()

        if df is not None:
            dataset_loaded = True
            n_cities = detect_n_cities(df)
            instance_ids = get_instance_ids(df)

            st.success(f" Source: {source_name}")
            st.caption(f"{len(df)} instances • {n_cities} cities each")

            selected_id = st.selectbox(
                "Select Instance:",
                instance_ids,
                help="Choose a specific TSP instance from the dataset."
            )
        else:
            st.warning(" Dataset not available. Use 'Random Cities' mode instead.")
            st.caption("Run `python data/download_dataset.py` to download.")

    else:
        n_cities = st.slider(
            "Number of Cities:",
            min_value=5, max_value=50, value=20, step=1,
            help="More cities = harder problem = bigger difference between algorithms."
        )
        random_seed = st.number_input(
            "Random Seed:",
            min_value=0, max_value=9999, value=42,
            help="Set a seed for reproducible city layouts."
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Algorithm Parameters ──
    st.markdown("###  Algorithm Parameters")

    with st.expander(" Hill Climbing", expanded=True):
        hc_max_iterations = st.slider(
            "Max Iterations (HC):",
            min_value=100, max_value=10000, value=5000, step=100,
            help="Safety limit. HC typically stops earlier at a local optimum."
        )

    with st.expander(" Simulated Annealing", expanded=True):
        sa_initial_temp = st.slider(
            "Initial Temperature:",
            min_value=100.0, max_value=50000.0, value=10000.0, step=500.0,
            help="Higher = more exploration at the start."
        )
        sa_cooling_rate = st.slider(
            "Cooling Rate:",
            min_value=0.990, max_value=0.99999, value=0.9995,
            step=0.0001, format="%.5f",
            help="Closer to 1 = slower cooling = more thorough search."
        )
        sa_max_iterations = st.slider(
            "Max Iterations (SA):",
            min_value=10000, max_value=500000, value=100000, step=10000,
            help="Total iterations budget for SA."
        )
        sa_min_temp = st.number_input(
            "Min Temperature:",
            min_value=1e-12, max_value=1.0, value=1e-8,
            format="%.1e",
            help="Stop when temperature drops below this threshold."
        )

    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    # ── Run Button ──
    run_single = st.button(
        " Run Optimization",
        use_container_width=True,
        type="primary",
    )


# ══════════════════════════════════════════════════════════════════════
#  Prepare City Data
# ══════════════════════════════════════════════════════════════════════

def get_cities() -> np.ndarray | None:
    """Get city coordinates from the selected data source."""
    if data_mode == " Kaggle / Sample Dataset" and dataset_loaded:
        row = get_instance_by_id(df, selected_id)
        return parse_instance(row, n_cities)
    elif data_mode == " Random Cities":
        return generate_random_cities(n_cities, seed=random_seed)
    return None


# ══════════════════════════════════════════════════════════════════════
#  Main Content — Tabbed Layout
# ══════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    " Route Comparison",
    " Convergence Analysis",
    " Batch Experiment",
])


# ── Helper: render metric card ──
def metric_card(label: str, value: str, sublabel: str = "", color: str = "#fff"):
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-sublabel">{sublabel}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════
#  Tab 1: Route Comparison
# ══════════════════════════════════════════════════════════════════════

with tab1:
    if run_single:
        cities = get_cities()
        if cities is None:
            st.error(" No city data available. Select a dataset or use random mode.")
        else:
            # Generate a shared initial route for fair comparison
            shared_route = generate_initial_route(len(cities))
            initial_distance = calculate_total_distance(shared_route, cities)

            st.markdown(f"""
            <div class="info-box">
                 Both algorithms start from the <b>same random initial route</b>
                (distance: <b>{initial_distance:.2f}</b>) for a fair comparison.
            </div>
            """, unsafe_allow_html=True)

            # ── Run both algorithms ──
            col_prog1, col_prog2 = st.columns(2)

            with col_prog1:
                with st.spinner(" Running Hill Climbing..."):
                    hc_result = hill_climbing(
                        cities, initial_route=shared_route,
                        max_iterations=hc_max_iterations,
                    )

            with col_prog2:
                with st.spinner(" Running Simulated Annealing..."):
                    sa_result = simulated_annealing(
                        cities, initial_route=shared_route,
                        initial_temperature=sa_initial_temp,
                        cooling_rate=sa_cooling_rate,
                        min_temperature=sa_min_temp,
                        max_iterations=sa_max_iterations,
                    )

            # Save results to session state for other tabs
            st.session_state["hc_result"] = hc_result
            st.session_state["sa_result"] = sa_result
            st.session_state["cities"] = cities
            st.session_state["initial_distance"] = initial_distance

            # ── Winner determination ──
            hc_dist = hc_result["best_distance"]
            sa_dist = sa_result["best_distance"]
            diff_pct = abs(hc_dist - sa_dist) / max(hc_dist, sa_dist) * 100

            if sa_dist < hc_dist:
                winner = "Simulated Annealing"
                winner_class = "winner-sa"
                winner_emoji = ""
            elif hc_dist < sa_dist:
                winner = "Hill Climbing"
                winner_class = "winner-hc"
                winner_emoji = ""
            else:
                winner = "Tie"
                winner_class = "winner-tie"
                winner_emoji = ""

            st.markdown(f"""
            <div style="text-align:center; margin: 1rem 0;">
                <span class="winner-badge {winner_class}">
                    {winner_emoji} Winner: {winner} — {diff_pct:.2f}% better
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Metric cards ──
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                hc_improve = (1 - hc_dist / initial_distance) * 100
                st.markdown(metric_card(
                    "HC Distance", f"{hc_dist:.2f}",
                    f"Improved {hc_improve:.1f}%", HC_COLOR
                ), unsafe_allow_html=True)
            with m2:
                sa_improve = (1 - sa_dist / initial_distance) * 100
                st.markdown(metric_card(
                    "SA Distance", f"{sa_dist:.2f}",
                    f"Improved {sa_improve:.1f}%", SA_COLOR
                ), unsafe_allow_html=True)
            with m3:
                st.markdown(metric_card(
                    "HC Time", f"{hc_result['execution_time']:.3f}s",
                    f"{hc_result['iterations']} iterations", HC_COLOR
                ), unsafe_allow_html=True)
            with m4:
                st.markdown(metric_card(
                    "SA Time", f"{sa_result['execution_time']:.3f}s",
                    f"{sa_result['iterations']} iterations", SA_COLOR
                ), unsafe_allow_html=True)

            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

            # ── Side-by-side route plots ──
            col1, col2 = st.columns(2)
            with col1:
                fig_hc = plot_route(
                    cities, hc_result["best_route"],
                    title=" Hill Climbing Route",
                    color=HC_COLOR,
                    distance=hc_dist,
                )
                st.plotly_chart(fig_hc, use_container_width=True, key="hc_route")

            with col2:
                fig_sa = plot_route(
                    cities, sa_result["best_route"],
                    title=" Simulated Annealing Route",
                    color=SA_COLOR,
                    distance=sa_dist,
                )
                st.plotly_chart(fig_sa, use_container_width=True, key="sa_route")

            # ── Algorithm details expander ──
            with st.expander(" Algorithm Details", expanded=False):
                det1, det2 = st.columns(2)
                with det1:
                    st.markdown(f"""
                    ** Hill Climbing**
                    - **Strategy:** Steepest-descent local search
                    - **Termination:** `{hc_result['terminated_reason']}`
                    - **Total Iterations:** {hc_result['iterations']}
                    - **Route:** `{list(hc_result['best_route'][:10])}...`
                    """)
                with det2:
                    st.markdown(f"""
                    ** Simulated Annealing**
                    - **Strategy:** Metropolis criterion + geometric cooling
                    - **Termination:** `{sa_result['terminated_reason']}`
                    - **Total Iterations:** {sa_result['iterations']}
                    - **Worse solutions accepted:** {sa_result['accepted_worse']}
                    - **Final Temperature:** {sa_result['final_temperature']:.2e}
                    """)

    else:
        # Show placeholder instructions when not yet run
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #888;">
            <h2 style="color:#555;"> Configure & Run</h2>
            <p>Select a data source, tune the parameters in the sidebar,<br>
            then click <b> Run Optimization</b> to compare both algorithms.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Tab 2: Convergence Analysis
# ══════════════════════════════════════════════════════════════════════

with tab2:
    if "hc_result" in st.session_state and "sa_result" in st.session_state:
        hc_result = st.session_state["hc_result"]
        sa_result = st.session_state["sa_result"]
        cities = st.session_state["cities"]

        st.markdown("###  Distance Convergence")
        st.markdown("""
        <div class="info-box">
            The convergence plot shows how each algorithm reduces the total tour distance
            over iterations. <span class="algo-tag-hc">Hill Climbing</span> typically
            plateaus quickly at a local optimum, while
            <span class="algo-tag-sa">Simulated Annealing</span> may initially accept
            worse solutions but explores more broadly, often finding better final solutions.
        </div>
        """, unsafe_allow_html=True)

        # Convergence overlay
        fig_conv = plot_convergence(hc_result["history"], sa_result["history"])
        st.plotly_chart(fig_conv, use_container_width=True, key="convergence")

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Temperature and comparison bar charts
        conv_col1, conv_col2 = st.columns(2)

        with conv_col1:
            st.markdown("###  SA Temperature Decay")
            fig_temp = plot_temperature_decay(sa_result["history"])
            st.plotly_chart(fig_temp, use_container_width=True, key="temp_decay")

        with conv_col2:
            st.markdown("###  Performance Metrics")
            fig_bar = plot_comparison_bar(hc_result, sa_result)
            st.plotly_chart(fig_bar, use_container_width=True, key="comparison_bar")

        # Analysis summary
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("###  Analysis Summary")

        initial_dist = st.session_state.get("initial_distance", 0)
        hc_dist = hc_result["best_distance"]
        sa_dist = sa_result["best_distance"]

        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.markdown(f"""
            **Convergence Behavior:**
            -  **Hill Climbing** converged in **{hc_result['iterations']}** iterations
              and found a local optimum of **{hc_dist:.2f}**
            -  **Simulated Annealing** ran for **{sa_result['iterations']}** iterations,
              accepting **{sa_result['accepted_worse']}** worse solutions
              along the way, ultimately finding distance **{sa_dist:.2f}**
            """)

        with analysis_col2:
            if sa_dist < hc_dist:
                gap = hc_dist - sa_dist
                st.markdown(f"""
                **Key Insight:**
                SA outperformed HC by **{gap:.2f}** distance units
                (**{gap/hc_dist*100:.1f}%** improvement). This demonstrates SA's
                ability to escape local optima through probabilistic acceptance
                of worse solutions during the high-temperature exploration phase.
                """)
            elif hc_dist < sa_dist:
                gap = sa_dist - hc_dist
                st.markdown(f"""
                **Key Insight:**
                HC found a better solution by **{gap:.2f}** distance units.
                This can happen when the problem landscape is smooth with few
                local optima, or when SA parameters need further tuning
                (try higher initial temperature or slower cooling rate).
                """)
            else:
                st.markdown("""
                **Key Insight:**
                Both algorithms found the same solution! This suggests the
                problem instance has a dominant local optimum that's easy to find.
                """)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem; color: #888;">
            <h2 style="color:#555;"> No Results Yet</h2>
            <p>Run an optimization from the <b> Route Comparison</b> tab first.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Tab 3: Batch Experiment
# ══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("###  Batch Experiment Mode")
    st.markdown("""
    <div class="info-box">
        Run both algorithms on <b>multiple TSP instances</b> to get statistically
        meaningful comparisons. This reveals which algorithm is consistently
        better across different problem configurations.
    </div>
    """, unsafe_allow_html=True)

    batch_col1, batch_col2 = st.columns([1, 1])

    with batch_col1:
        batch_source = st.radio(
            "Batch data source:",
            ["Dataset Instances", "Random Instances"],
            key="batch_source",
        )

    with batch_col2:
        if batch_source == "Dataset Instances":
            if df is not None:
                max_batch = min(len(df), 100)
                batch_size = st.slider(
                    "Number of instances:", 5, max_batch, min(20, max_batch),
                    key="batch_size_ds"
                )
            else:
                st.warning("Dataset not loaded. Switch to Random Instances.")
                batch_size = 0
        else:
            batch_size = st.slider(
                "Number of random instances:", 5, 100, 20,
                key="batch_size_rand"
            )
            batch_n_cities = st.slider(
                "Cities per instance:", 5, 50, 20,
                key="batch_n_cities"
            )

    run_batch = st.button(
        " Run Batch Experiment",
        use_container_width=True,
        disabled=(batch_size == 0 if batch_source == "Dataset Instances" and df is None else False),
    )

    if run_batch and batch_size > 0:
        results = []
        progress_bar = st.progress(0, text="Running batch experiment...")

        for idx in range(batch_size):
            # Get cities for this instance
            if batch_source == "Dataset Instances" and df is not None:
                instance_ids = get_instance_ids(df)
                inst_id = instance_ids[idx % len(instance_ids)]
                row = get_instance_by_id(df, inst_id)
                batch_cities = parse_instance(row, detect_n_cities(df))
            else:
                batch_cities = generate_random_cities(
                    batch_n_cities, seed=idx * 42 + 7
                )
                inst_id = idx + 1

            # Shared initial route
            shared = generate_initial_route(len(batch_cities))
            init_dist = calculate_total_distance(shared, batch_cities)

            # Run both algorithms
            hc_res = hill_climbing(
                batch_cities, initial_route=shared, max_iterations=hc_max_iterations
            )
            sa_res = simulated_annealing(
                batch_cities, initial_route=shared,
                initial_temperature=sa_initial_temp,
                cooling_rate=sa_cooling_rate,
                min_temperature=sa_min_temp,
                max_iterations=sa_max_iterations,
            )

            # Determine winner
            if sa_res["best_distance"] < hc_res["best_distance"]:
                winner = "SA"
            elif hc_res["best_distance"] < sa_res["best_distance"]:
                winner = "HC"
            else:
                winner = "Tie"

            gap_pct = (hc_res["best_distance"] - sa_res["best_distance"]) / \
                      hc_res["best_distance"] * 100

            results.append({
                "Instance": inst_id,
                "Initial_Dist": round(init_dist, 2),
                "HC_Distance": round(hc_res["best_distance"], 2),
                "SA_Distance": round(sa_res["best_distance"], 2),
                "Winner": winner,
                "Gap_%": round(gap_pct, 2),
                "HC_Time": round(hc_res["execution_time"], 4),
                "SA_Time": round(sa_res["execution_time"], 4),
                "HC_Iters": hc_res["iterations"],
                "SA_Iters": sa_res["iterations"],
            })

            progress_bar.progress(
                (idx + 1) / batch_size,
                text=f"Instance {idx + 1}/{batch_size} completed..."
            )

        progress_bar.empty()
        results_df = pd.DataFrame(results)

        # ── Summary Statistics ──
        sa_wins = len(results_df[results_df["Winner"] == "SA"])
        hc_wins = len(results_df[results_df["Winner"] == "HC"])
        ties = len(results_df[results_df["Winner"] == "Tie"])

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Summary cards
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(metric_card(
                "SA Wins", f"{sa_wins}/{batch_size}",
                f"{sa_wins/batch_size*100:.0f}% win rate", SA_COLOR
            ), unsafe_allow_html=True)
        with s2:
            st.markdown(metric_card(
                "HC Wins", f"{hc_wins}/{batch_size}",
                f"{hc_wins/batch_size*100:.0f}% win rate", HC_COLOR
            ), unsafe_allow_html=True)
        with s3:
            avg_gap = results_df["Gap_%"].mean()
            st.markdown(metric_card(
                "Avg SA Advantage", f"{avg_gap:.2f}%",
                "Positive = SA better", "#FFD93D"
            ), unsafe_allow_html=True)
        with s4:
            st.markdown(metric_card(
                "Ties", str(ties),
                "Equal performance", "#888"
            ), unsafe_allow_html=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        # Visualization
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            fig_box = plot_batch_results(results_df)
            st.plotly_chart(fig_box, use_container_width=True, key="batch_box")

        with viz_col2:
            fig_scatter = plot_batch_scatter(results_df)
            st.plotly_chart(fig_scatter, use_container_width=True, key="batch_scatter")

        # Results table
        st.markdown("###  Detailed Results")

        # Style winner column
        def highlight_winner(val):
            if val == "SA":
                return f"color: {SA_COLOR}; font-weight: 600;"
            elif val == "HC":
                return f"color: {HC_COLOR}; font-weight: 600;"
            return "color: #FFD93D;"

        styled_df = results_df.style.map(
            highlight_winner, subset=["Winner"]
        ).format({
            "Initial_Dist": "{:.2f}",
            "HC_Distance": "{:.2f}",
            "SA_Distance": "{:.2f}",
            "Gap_%": "{:+.2f}%",
            "HC_Time": "{:.4f}s",
            "SA_Time": "{:.4f}s",
        })

        st.dataframe(styled_df, use_container_width=True, height=400)

        # Save to session for persistence
        st.session_state["batch_results"] = results_df

    elif "batch_results" in st.session_state and not run_batch:
        # Show previous batch results
        results_df = st.session_state["batch_results"]

        sa_wins = len(results_df[results_df["Winner"] == "SA"])
        hc_wins = len(results_df[results_df["Winner"] == "HC"])
        ties = len(results_df[results_df["Winner"] == "Tie"])
        batch_size_prev = len(results_df)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(metric_card(
                "SA Wins", f"{sa_wins}/{batch_size_prev}",
                f"{sa_wins/batch_size_prev*100:.0f}% win rate", SA_COLOR
            ), unsafe_allow_html=True)
        with s2:
            st.markdown(metric_card(
                "HC Wins", f"{hc_wins}/{batch_size_prev}",
                f"{hc_wins/batch_size_prev*100:.0f}% win rate", HC_COLOR
            ), unsafe_allow_html=True)
        with s3:
            avg_gap = results_df["Gap_%"].mean()
            st.markdown(metric_card(
                "Avg SA Advantage", f"{avg_gap:.2f}%",
                "Positive = SA better", "#FFD93D"
            ), unsafe_allow_html=True)
        with s4:
            st.markdown(metric_card(
                "Ties", str(ties),
                "Equal performance", "#888"
            ), unsafe_allow_html=True)

        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            fig_box = plot_batch_results(results_df)
            st.plotly_chart(fig_box, use_container_width=True, key="batch_box_cached")
        with viz_col2:
            fig_scatter = plot_batch_scatter(results_df)
            st.plotly_chart(fig_scatter, use_container_width=True, key="batch_scatter_cached")

        st.dataframe(results_df, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════
#  Footer
# ══════════════════════════════════════════════════════════════════════

st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 1rem; color: #555; font-size: 0.8rem;">
    TSP Optimizer • AI in Action — Applied Programming Project •
    Hill Climbing vs Simulated Annealing Comparison
</div>
""", unsafe_allow_html=True)
