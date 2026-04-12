"""
Hill Climbing Algorithm for TSP
================================
Implements the Steepest-Ascent Hill Climbing (or more precisely,
Steepest-Descent, since we minimize distance) algorithm for the
Traveling Salesman Problem.

Hill Climbing is a local search algorithm that:
1. Starts with an initial solution (random route)
2. Evaluates ALL neighbors (2-opt swaps)
3. Moves to the BEST improving neighbor
4. Repeats until no improving neighbor exists (local optimum)

Key Characteristic: Greedy — always picks the best immediate improvement.
Weakness: Prone to getting stuck in local optima, never explores worse
solutions that could lead to better global solutions.
"""

import time
import numpy as np
from src.models.tsp import (
    calculate_total_distance,
    calculate_delta_distance,
    two_opt_swap,
    generate_initial_route,
)


def hill_climbing(
    cities: np.ndarray,
    initial_route: np.ndarray = None,
    max_iterations: int = 5000,
    log_interval: int = 1,
) -> dict:
    """
    Solve TSP using Steepest-Ascent Hill Climbing with 2-opt neighborhood.

    At each iteration, the algorithm examines ALL possible 2-opt swaps
    and selects the one that yields the greatest distance reduction.
    The search terminates when no swap improves the current solution
    (i.e., a local optimum is reached).

    Args:
        cities: 2D array of (x, y) coordinates with shape (n_cities, 2).
        initial_route: Starting route permutation. If None, a random
                       route is generated.
        max_iterations: Maximum number of iterations (safety limit).
        log_interval: Record history every N iterations for visualization.

    Returns:
        Dictionary containing:
            - 'best_route': The best route found (array of city indices).
            - 'best_distance': Total distance of the best route.
            - 'history': List of (iteration, distance) tuples for convergence plot.
            - 'iterations': Total iterations performed.
            - 'execution_time': Wall-clock time in seconds.
            - 'algorithm': Algorithm name string.
            - 'terminated_reason': Why the search stopped.
    """
    n_cities = len(cities)
    start_time = time.time()

    # Initialize route
    if initial_route is not None:
        current_route = initial_route.copy()
    else:
        current_route = generate_initial_route(n_cities)

    current_distance = calculate_total_distance(current_route, cities)
    history = [(0, float(current_distance))]

    best_route = current_route.copy()
    best_distance = current_distance
    terminated_reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        # --- Steepest Descent: find the best 2-opt swap ---
        best_delta = 0.0
        best_i, best_j = -1, -1

        for i in range(1, n_cities - 1):
            for j in range(i + 1, n_cities):
                delta = calculate_delta_distance(current_route, cities, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j

        # If no improving swap found, we've reached a local optimum
        if best_delta >= 0:
            terminated_reason = "local_optimum_reached"
            break

        # Apply the best swap
        current_route = two_opt_swap(current_route, best_i, best_j)
        current_distance += best_delta

        # Track the global best
        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_route.copy()

        # Log history for convergence plotting
        if iteration % log_interval == 0:
            history.append((iteration, float(current_distance)))

    # Ensure the final state is logged
    if history[-1][0] != iteration:
        history.append((iteration, float(current_distance)))

    execution_time = time.time() - start_time

    return {
        "best_route": best_route,
        "best_distance": float(best_distance),
        "history": history,
        "iterations": iteration,
        "execution_time": execution_time,
        "algorithm": "Hill Climbing",
        "terminated_reason": terminated_reason,
    }
