#  TSP Optimizer — Hill Climbing vs Simulated Annealing

> **AI in Action — Applied Programming Project**
>
> An interactive optimization simulation comparing **Hill Climbing** and **Simulated Annealing** algorithms on the **Traveling Salesman Problem (TSP)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

##  Table of Contents

- [Overview](#overview)
- [Algorithms](#algorithms)
  - [Hill Climbing](#hill-climbing)
  - [Simulated Annealing](#simulated-annealing)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results & Analysis](#results--analysis)

---

## Overview

The **Traveling Salesman Problem (TSP)** is a classic NP-hard combinatorial optimization problem: given a set of cities, find the shortest possible route that visits each city exactly once and returns to the starting city.

This project implements and compares two optimization approaches:
1. **Hill Climbing** — a greedy local search that always moves to the best neighbor
2. **Simulated Annealing** — a metaheuristic that probabilistically accepts worse solutions to escape local optima

The application provides an interactive **Streamlit** dashboard for visualizing routes, analyzing convergence behavior, and running batch experiments.

---

## Algorithms

### Hill Climbing

Hill Climbing is a **local search algorithm** that iteratively improves a solution by examining neighboring solutions and moving to the best one. It uses the **2-opt swap** operator to generate neighbors.

**How it works:**
1. Start with a random initial route
2. Examine all possible 2-opt swaps (reverse a segment of the route)
3. Accept the swap that gives the largest distance reduction
4. Repeat until no improving swap exists → **local optimum reached**

**Strengths:**
- Simple and fast
- Guaranteed to find a local optimum

**Weaknesses:**
- Gets **stuck in local optima** — cannot escape once it reaches one
- Solution quality depends heavily on the starting point

```
Iteration →  ●───●───●───●───■  ← Local Optimum (stuck!)
Distance ↓                        (may not be globally optimal)
```

### Simulated Annealing

Simulated Annealing is a **metaheuristic** inspired by the annealing process in metallurgy. It introduces controlled randomness to explore the solution space more broadly.

**How it works:**
1. Start with a random initial route and a high **temperature** T
2. Randomly select a 2-opt swap
3. If the swap **improves** the solution → always accept
4. If the swap **worsens** the solution → accept with probability `exp(-Δ/T)`
5. Reduce temperature: `T = T × cooling_rate`
6. Repeat until temperature drops below threshold

**Metropolis Acceptance Criterion:**
```
P(accept worse solution) = exp(-ΔDistance / Temperature)
```

At **high temperature**: almost any move is accepted → **exploration**
At **low temperature**: only improvements accepted → **exploitation**

**Strengths:**
- Can **escape local optima** by accepting worse solutions early on
- Often finds **near-global optimal** solutions
- Balances exploration and exploitation

**Weaknesses:**
- Requires parameter tuning (temperature, cooling rate)
- Slower than Hill Climbing due to more iterations

```
Iteration →  ●───●↗●↘●───●─↗─●───●───●  ← Better Final Solution
Distance ↓           ↑ accepts worse        (escaped local optimum!)
                     ↑ temporarily
```

---

## Features

| Feature | Description |
|---------|-------------|
|  **Route Comparison** | Side-by-side interactive route maps for both algorithms |
|  **Convergence Analysis** | Overlaid distance curves showing optimization progress |
|  **Temperature Decay** | Visualization of SA's cooling schedule |
|  **Batch Experiment** | Run both algorithms on multiple instances with statistical analysis |
|  **Kaggle Dataset** | Uses TSPLIB dataset (2,783 instances, 20 cities each) |
|  **Random Generator** | Custom city generation (5-50 cities) for additional experiments |
| ⚡ **Numba JIT** | JIT-compiled distance calculations for high performance |
| 🎨 **Premium UI** | Dark-themed interactive dashboard with Plotly charts |

---

## Project Structure

```
traveling_salesman/
├── app.py                          # Streamlit main entry point
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore rules
├── data/
│   ├── download_dataset.py         # Kaggle dataset downloader script
│   └── sample/
│       └── sample_tsp.csv          # Bundled sample data (20 instances)
├── src/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── hill_climbing.py        # Hill Climbing implementation
│   │   └── simulated_annealing.py  # Simulated Annealing implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── tsp.py                  # TSP core model (distance, 2-opt, routes)
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py          # Data loading with Kaggle/fallback
│       └── visualization.py        # Plotly visualization functions
└── assets/                         # UI assets
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd traveling_salesman
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **(Optional) Download Kaggle dataset:**
```bash
# First, set up Kaggle API credentials:
# https://github.com/Kaggle/kagglehub#authenticate
python data/download_dataset.py
```

> **Note:** The app works without Kaggle credentials using the bundled sample CSV or random city generator.

---

## Usage

### Launch the Application
```bash
streamlit run app.py
```

### Application Tabs

**Tab 1 —  Route Comparison:**
- Select a data source (Kaggle dataset, sample CSV, or random cities)
- Configure algorithm parameters in the sidebar
- Click " Run Optimization" to see side-by-side route visualizations
- View metric cards for distance, time, and improvement percentages

**Tab 2 —  Convergence Analysis:**
- Examine how each algorithm's solution improves over iterations
- View SA's temperature decay schedule
- Read auto-generated analysis comparing the two approaches

**Tab 3 —  Batch Experiment:**
- Run both algorithms on multiple instances simultaneously
- View statistical distributions via box plots
- Analyze per-instance HC vs SA scatter comparison
- See win rates, average advantage, and detailed results table

---

## Dataset

### Primary: Kaggle TSPLIB Dataset
- **Source:** [ziya07/traveling-salesman-problem-tsplib-dataset](https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset)
- **Instances:** 2,783 TSP instances
- **Cities:** 20 per instance, randomly distributed in 2D space
- **Columns:** Instance_ID, City_1_X, City_1_Y, ..., City_20_X, City_20_Y, Total_Distance, Best_Route

### Fallback: Bundled Sample CSV
- 20 sample instances included in `data/sample/sample_tsp.csv`
- Same format as the Kaggle dataset

### Custom: Random City Generator
- Generate 5–50 cities with configurable random seed
- Useful for controlled experiments

---

## Results & Analysis

### Typical Findings

| Metric | Hill Climbing | Simulated Annealing |
|--------|:---:|:---:|
| **Solution Quality** | Good (local optimum) | Better (near-global optimum) |
| **Speed** | Faster | Slower (more iterations) |
| **Local Optima** | Gets stuck | Escapes via probabilistic acceptance |
| **Consistency** | Variable | More consistent across runs |
| **Parameter Tuning** | Minimal | Requires tuning (temperature, cooling) |

### Key Observations
1. **SA typically outperforms HC** in finding shorter routes, especially for larger instances
2. **HC converges faster** but to suboptimal solutions
3. **SA's advantage increases** with more cities (more complex landscapes have more local optima to escape)
4. The **cooling rate** is the most critical SA parameter — values closer to 1.0 give better results but take longer

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive web dashboard |
| **NumPy** | Numerical computations |
| **Numba** | JIT compilation for performance |
| **Plotly** | Interactive visualization |
| **Pandas** | Data handling |
| **KaggleHub** | Dataset management |

---


