"""
Simulated Annealing Algorithm for TSP
======================================
Implements the Simulated Annealing metaheuristic for the Traveling
Salesman Problem, inspired by the metallurgical annealing process.

Simulated Annealing (SA) works by:
1. Starting with an initial solution and a high "temperature"
2. At each step, generating a random neighbor (2-opt swap)
3. Always accepting improvements
4. Accepting WORSE solutions with probability exp(-delta / temperature)
5. Gradually reducing the temperature (cooling schedule)
6. As temperature cools, acceptance of worse solutions decreases

Key Advantage over Hill Climbing:
    By accepting worse solutions probabilistically, SA can ESCAPE local
    optima. At high temperatures it explores broadly; at low temperatures
    it converges to a refined solution — balancing exploration vs exploitation.
"""

import time
import math
import numpy as np
from src.models.tsp import (
    calculate_total_distance,
    calculate_delta_distance,
    two_opt_swap,
    generate_initial_route,
)


def simulated_annealing(
    cities: np.ndarray,
    initial_route: np.ndarray = None,
    initial_temperature: float = 10000.0,
    cooling_rate: float = 0.9995,
    min_temperature: float = 1e-8,
    max_iterations: int = 100000,
    log_interval: int = 100,
) -> dict:
    """
    Solve TSP using Simulated Annealing with 2-opt random neighborhood.

    The algorithm randomly selects a 2-opt swap at each iteration.
    Better solutions are always accepted. Worse solutions are accepted
    with probability exp(-delta_distance / temperature), which decreases
    as the temperature cools according to the geometric cooling schedule:
        T_new = T_old * cooling_rate

    Args:
        cities: 2D array of (x, y) coordinates with shape (n_cities, 2).
        initial_route: Starting route permutation. If None, a random
                       route is generated.
        initial_temperature: Starting temperature. Higher values mean
                             more exploration early on.
        cooling_rate: Multiplicative factor for geometric cooling (0 < r < 1).
                      Values closer to 1 cool more slowly (more thorough search).
        min_temperature: Temperature threshold to stop the algorithm.
        max_iterations: Maximum iterations (safety limit).
        log_interval: Record history every N iterations.

    Returns:
        Dictionary containing:
            - 'best_route': The best route found across the entire run.
            - 'best_distance': Total distance of the best route.
            - 'history': List of (iteration, current_distance, temperature) tuples.
            - 'iterations': Total iterations performed.
            - 'execution_time': Wall-clock time in seconds.
            - 'algorithm': Algorithm name string.
            - 'terminated_reason': Why the search stopped.
            - 'accepted_worse': Count of times a worse solution was accepted.
            - 'final_temperature': Temperature when the algorithm stopped.
    """
    n_cities = len(cities)
    start_time = time.time()

    # Initialize route
    if initial_route is not None:
        current_route = initial_route.copy()
    else:
        current_route = generate_initial_route(n_cities)

    current_distance = calculate_total_distance(current_route, cities)

    # Track the overall best solution found during the entire run
    best_route = current_route.copy()
    best_distance = current_distance

    temperature = initial_temperature
    history = [(0, float(current_distance), temperature)]
    accepted_worse = 0
    terminated_reason = "max_iterations_reached"

    for iteration in range(1, max_iterations + 1):
        # Check temperature stopping criterion
        if temperature < min_temperature:
            terminated_reason = "temperature_cooled"
            break

        # Generate random neighbor via 2-opt swap
        i = np.random.randint(1, n_cities - 1)
        j = np.random.randint(i + 1, n_cities)

        # Efficient O(1) delta calculation instead of full tour recalculation
        delta = calculate_delta_distance(current_route, cities, i, j)

        # --- Metropolis Acceptance Criterion ---
        # If delta < 0: improvement -> always accept
        # If delta >= 0: worse solution -> accept with probability exp(-delta/T)
        if delta < 0:
            accept = True
        else:
            acceptance_probability = math.exp(-delta / temperature)
            accept = np.random.random() < acceptance_probability

        if accept:
            current_route = two_opt_swap(current_route, i, j)
            current_distance += delta

            if delta >= 0:
                accepted_worse += 1

            # Update global best if this is the best we've ever seen
            if current_distance < best_distance:
                best_distance = current_distance
                best_route = current_route.copy()

        # Geometric cooling schedule: T = T * alpha
        temperature *= cooling_rate

        # Log history for convergence and temperature plots
        if iteration % log_interval == 0:
            history.append((iteration, float(current_distance), temperature))

    # Ensure the final state is logged
    final_iter = iteration if iteration <= max_iterations else max_iterations
    if history[-1][0] != final_iter:
        history.append((final_iter, float(current_distance), temperature))

    execution_time = time.time() - start_time

    return {
        "best_route": best_route,
        "best_distance": float(best_distance),
        "history": history,
        "iterations": final_iter,
        "execution_time": execution_time,
        "algorithm": "Simulated Annealing",
        "terminated_reason": terminated_reason,
        "accepted_worse": accepted_worse,
        "final_temperature": temperature,
    }
