"""
Visualization Module
====================
Plotly-based chart generation functions for TSP route visualization,
convergence analysis, and algorithm comparison.

All functions return Plotly Figure objects that can be rendered
directly in Streamlit using st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ──────────────────────────── Color Palette ────────────────────────────
# Premium dark theme colors
HC_COLOR = "#FF6B6B"       # Coral red for Hill Climbing
SA_COLOR = "#4ECDC4"       # Teal for Simulated Annealing
BG_COLOR = "rgba(0,0,0,0)" # Transparent for Streamlit dark mode
GRID_COLOR = "rgba(255,255,255,0.08)"
TEXT_COLOR = "#E0E0E0"
CITY_COLOR = "#FFD93D"     # Gold for city markers
START_COLOR = "#6BCB77"    # Green for start city


def _base_layout(title: str = "", height: int = 500) -> dict:
    """
    Common Plotly layout settings for consistent dark-theme styling.

    Args:
        title: Chart title.
        height: Chart height in pixels.

    Returns:
        Dictionary of layout properties.
    """
    return dict(
        title=dict(text=title, font=dict(size=18, color=TEXT_COLOR)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, sans-serif", color=TEXT_COLOR),
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color=TEXT_COLOR),
        ),
    )


def plot_route(
    cities: np.ndarray,
    route: np.ndarray,
    title: str = "TSP Route",
    color: str = HC_COLOR,
    distance: float = None,
) -> go.Figure:
    """
    Plot a TSP route as a 2D map with city markers and connecting edges.

    Displays cities as dots with numbered labels, connected by lines
    following the route order. The start city is highlighted in green.

    Args:
        cities: 2D array of (x, y) coordinates.
        route: Array of city indices defining visit order.
        title: Chart title.
        color: Color for route lines.
        distance: Optional total distance to show in subtitle.

    Returns:
        Plotly Figure object.
    """
    # Build ordered coordinate lists (close the loop)
    ordered = np.append(route, route[0])
    x_route = cities[ordered, 0]
    y_route = cities[ordered, 1]

    fig = go.Figure()

    # Route lines
    fig.add_trace(go.Scatter(
        x=x_route, y=y_route,
        mode="lines",
        line=dict(color=color, width=2.5),
        name="Route",
        opacity=0.7,
    ))

    # City markers (all cities)
    fig.add_trace(go.Scatter(
        x=cities[:, 0], y=cities[:, 1],
        mode="markers+text",
        marker=dict(size=12, color=CITY_COLOR, symbol="circle",
                    line=dict(width=1.5, color="rgba(0,0,0,0.5)")),
        text=[str(i) for i in range(len(cities))],
        textposition="top center",
        textfont=dict(size=9, color=TEXT_COLOR),
        name="Cities",
    ))

    # Highlight start city
    start = route[0]
    fig.add_trace(go.Scatter(
        x=[cities[start, 0]], y=[cities[start, 1]],
        mode="markers",
        marker=dict(size=18, color=START_COLOR, symbol="star",
                    line=dict(width=2, color="white")),
        name="Start City",
    ))

    # Build title with distance info
    subtitle = f"<br><span style='font-size:13px; color:#aaa;'>Distance: {distance:.2f}</span>" if distance else ""

    layout = _base_layout(title=f"{title}{subtitle}")
    layout["xaxis"]["title"] = "X Coordinate"
    layout["yaxis"]["title"] = "Y Coordinate"
    layout["showlegend"] = True
    fig.update_layout(**layout)

    return fig


def plot_convergence(
    hc_history: list,
    sa_history: list,
    title: str = "Convergence Comparison",
) -> go.Figure:
    """
    Overlay convergence curves for Hill Climbing and Simulated Annealing.

    Shows how the objective function (total distance) changes over
    iterations for both algorithms on the same chart.

    Args:
        hc_history: Hill Climbing history as [(iteration, distance), ...].
        sa_history: SA history as [(iteration, distance, temp), ...].
        title: Chart title.

    Returns:
        Plotly Figure with overlaid convergence lines.
    """
    fig = go.Figure()

    # Hill Climbing convergence
    hc_iters = [h[0] for h in hc_history]
    hc_dists = [h[1] for h in hc_history]
    fig.add_trace(go.Scatter(
        x=hc_iters, y=hc_dists,
        mode="lines",
        name="Hill Climbing",
        line=dict(color=HC_COLOR, width=2.5),
    ))

    # Simulated Annealing convergence
    sa_iters = [h[0] for h in sa_history]
    sa_dists = [h[1] for h in sa_history]
    fig.add_trace(go.Scatter(
        x=sa_iters, y=sa_dists,
        mode="lines",
        name="Simulated Annealing",
        line=dict(color=SA_COLOR, width=2.5),
    ))

    layout = _base_layout(title=title)
    layout["xaxis"]["title"] = "Iteration"
    layout["yaxis"]["title"] = "Total Distance"
    fig.update_layout(**layout)

    return fig


def plot_temperature_decay(sa_history: list) -> go.Figure:
    """
    Plot the temperature decay curve for Simulated Annealing.

    Visualizes how the temperature decreases over iterations,
    showing the transition from exploration to exploitation.

    Args:
        sa_history: SA history as [(iteration, distance, temperature), ...].

    Returns:
        Plotly Figure with temperature curve.
    """
    iters = [h[0] for h in sa_history]
    temps = [h[2] for h in sa_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=temps,
        mode="lines",
        name="Temperature",
        line=dict(color="#FF9F43", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(255,159,67,0.15)",
    ))

    layout = _base_layout(title="SA Temperature Decay Schedule")
    layout["xaxis"]["title"] = "Iteration"
    layout["yaxis"]["title"] = "Temperature"
    layout["yaxis"]["type"] = "log"
    fig.update_layout(**layout)

    return fig


def plot_comparison_bar(hc_result: dict, sa_result: dict) -> go.Figure:
    """
    Create a grouped bar chart comparing key metrics of both algorithms.

    Compares: Final Distance, Execution Time, and Iterations Used.

    Args:
        hc_result: Hill Climbing result dictionary.
        sa_result: Simulated Annealing result dictionary.

    Returns:
        Plotly Figure with grouped bars.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Final Distance", "Execution Time (s)", "Iterations"],
        horizontal_spacing=0.12,
    )

    algorithms = ["Hill Climbing", "Simulated Annealing"]
    colors = [HC_COLOR, SA_COLOR]

    # Distance comparison
    distances = [hc_result["best_distance"], sa_result["best_distance"]]
    fig.add_trace(go.Bar(
        x=algorithms, y=distances, marker_color=colors,
        text=[f"{d:.1f}" for d in distances], textposition="outside",
        showlegend=False,
    ), row=1, col=1)

    # Time comparison
    times = [hc_result["execution_time"], sa_result["execution_time"]]
    fig.add_trace(go.Bar(
        x=algorithms, y=times, marker_color=colors,
        text=[f"{t:.3f}s" for t in times], textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    # Iterations comparison
    iters = [hc_result["iterations"], sa_result["iterations"]]
    fig.add_trace(go.Bar(
        x=algorithms, y=iters, marker_color=colors,
        text=[str(i) for i in iters], textposition="outside",
        showlegend=False,
    ), row=1, col=3)

    layout = _base_layout(title="Algorithm Performance Comparison", height=400)
    fig.update_layout(**layout)

    # Style subplot axes
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=GRID_COLOR, row=1, col=i)
        fig.update_yaxes(gridcolor=GRID_COLOR, row=1, col=i)

    return fig


def plot_batch_results(results_df) -> go.Figure:
    """
    Create a box plot comparing distance distributions from batch experiments.

    Args:
        results_df: DataFrame with columns: Instance_ID, HC_Distance, SA_Distance.

    Returns:
        Plotly Figure with box plots.
    """
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=results_df["HC_Distance"],
        name="Hill Climbing",
        marker_color=HC_COLOR,
        boxmean=True,
    ))

    fig.add_trace(go.Box(
        y=results_df["SA_Distance"],
        name="Simulated Annealing",
        marker_color=SA_COLOR,
        boxmean=True,
    ))

    layout = _base_layout(title="Distance Distribution — Batch Comparison", height=450)
    layout["yaxis"]["title"] = "Total Distance"
    fig.update_layout(**layout)

    return fig


def plot_batch_scatter(results_df) -> go.Figure:
    """
    Scatter plot of HC distance vs SA distance per instance.
    Points below the diagonal line indicate SA outperformed HC.

    Args:
        results_df: DataFrame with HC_Distance and SA_Distance columns.

    Returns:
        Plotly Figure with scatter comparison.
    """
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=results_df["HC_Distance"],
        y=results_df["SA_Distance"],
        mode="markers",
        marker=dict(
            size=8,
            color=results_df["SA_Distance"] - results_df["HC_Distance"],
            colorscale=[[0, SA_COLOR], [0.5, "#888"], [1, HC_COLOR]],
            colorbar=dict(title="SA - HC"),
            line=dict(width=1, color="white"),
        ),
        name="Instances",
        hovertemplate=(
            "HC: %{x:.1f}<br>SA: %{y:.1f}<br>"
            "Diff: %{customdata:.1f}<extra></extra>"
        ),
        customdata=results_df["SA_Distance"] - results_df["HC_Distance"],
    ))

    # Diagonal line (equal performance)
    max_val = max(
        results_df["HC_Distance"].max(),
        results_df["SA_Distance"].max()
    ) * 1.05
    min_val = min(
        results_df["HC_Distance"].min(),
        results_df["SA_Distance"].min()
    ) * 0.95

    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
        name="Equal Line",
        showlegend=True,
    ))

    layout = _base_layout(title="HC vs SA — Per Instance Comparison", height=500)
    layout["xaxis"]["title"] = "Hill Climbing Distance"
    layout["yaxis"]["title"] = "Simulated Annealing Distance"
    fig.update_layout(**layout)

    # Add annotation explaining the diagonal
    fig.add_annotation(
        text="Below line = SA better",
        x=max_val * 0.95, y=min_val + (max_val - min_val) * 0.1,
        showarrow=False,
        font=dict(size=11, color=SA_COLOR),
    )
    fig.add_annotation(
        text="Above line = HC better",
        x=min_val + (max_val - min_val) * 0.1, y=max_val * 0.95,
        showarrow=False,
        font=dict(size=11, color=HC_COLOR),
    )

    return fig
