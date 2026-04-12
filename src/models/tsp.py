"""
TSP Problem Model
=================
Core data structures and utility functions for the Traveling Salesman Problem.
Provides distance calculations, route generation, and neighborhood operators
optimized with Numba JIT compilation for high performance.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _euclidean_distance(city_a: np.ndarray, city_b: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two cities.

    Args:
        city_a: NumPy array [x, y] for city A.
        city_b: NumPy array [x, y] for city B.

    Returns:
        Euclidean distance as a float.
    """
    dx = city_a[0] - city_b[0]
    dy = city_a[1] - city_b[1]
    return np.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def calculate_total_distance(route: np.ndarray, cities: np.ndarray) -> float:
    """
    Calculate the total tour distance for a given route through all cities.
    The tour is a closed loop — it returns to the starting city.

    Args:
        route: 1D array of city indices defining the visit order.
        cities: 2D array of shape (n_cities, 2) with (x, y) coordinates.

    Returns:
        Total Euclidean distance of the complete tour.
    """
    total = 0.0
    n = len(route)
    for i in range(n):
        total += _euclidean_distance(cities[route[i]], cities[route[(i + 1) % n]])
    return total


@njit(cache=True)
def two_opt_swap(route: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Perform a 2-opt swap: reverse the segment between indices i and j.
    This is the primary neighborhood operator for TSP local search.

    The 2-opt swap removes two edges from the tour and reconnects
    the resulting path segments in the only other possible way,
    potentially uncrossing edges and reducing total distance.

    Args:
        route: Current route as a 1D array of city indices.
        i: Start index of the segment to reverse.
        j: End index of the segment to reverse.

    Returns:
        New route with the segment [i, j] reversed.
    """
    new_route = route.copy()
    new_route[i:j + 1] = new_route[i:j + 1][::-1]
    return new_route


@njit(cache=True)
def calculate_delta_distance(
    route: np.ndarray, cities: np.ndarray, i: int, j: int
) -> float:
    """
    Calculate the change in total distance from a 2-opt swap between
    positions i and j WITHOUT computing the full tour distance.
    This is an O(1) operation instead of O(n), providing significant
    speedup for large instances.

    The delta is computed by comparing the two removed edges with the
    two newly created edges after the swap.

    Args:
        route: Current route.
        cities: City coordinates.
        i: Start index of the segment to reverse.
        j: End index of the segment to reverse.

    Returns:
        Change in distance (negative means improvement).
    """
    n = len(route)
    # Nodes involved in the swap edges
    a, b = route[i - 1], route[i]
    c, d = route[j], route[(j + 1) % n]

    # Old edges: (a -> b) and (c -> d)
    old_dist = _euclidean_distance(cities[a], cities[b]) + \
               _euclidean_distance(cities[c], cities[d])

    # New edges: (a -> c) and (b -> d) after reversal
    new_dist = _euclidean_distance(cities[a], cities[c]) + \
               _euclidean_distance(cities[b], cities[d])

    return new_dist - old_dist


def generate_initial_route(n_cities: int) -> np.ndarray:
    """
    Generate a random initial route (permutation of city indices).

    Args:
        n_cities: Number of cities in the TSP instance.

    Returns:
        Random permutation of city indices as a NumPy array.
    """
    route = np.arange(n_cities)
    np.random.shuffle(route)
    return route


def generate_random_cities(n_cities: int, x_range: tuple = (0, 100),
                           y_range: tuple = (0, 100),
                           seed: int = None) -> np.ndarray:
    """
    Generate random city coordinates on a 2D plane.

    Args:
        n_cities: Number of cities to generate.
        x_range: Tuple (min_x, max_x) for the x-coordinate range.
        y_range: Tuple (min_y, max_y) for the y-coordinate range.
        seed: Optional random seed for reproducibility.

    Returns:
        2D NumPy array of shape (n_cities, 2) with (x, y) coordinates.
    """
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(x_range[0], x_range[1], n_cities)
    y = np.random.uniform(y_range[0], y_range[1], n_cities)
    return np.column_stack((x, y))
