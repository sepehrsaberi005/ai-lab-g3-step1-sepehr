"""
Urban Route Finding Simulation - AI Lab Session 2
================================================
This script adapts and enhances the class vacuum-world simulation code 
for the Urban Route Finding project.

It models an urban street grid with intersections, traffic obstacles,
and compares three types of navigation agents across 100 randomized trials:
  1. Random Route Agent (Random Walk)
  2. Reflex Route Agent (Greedy Local Distance)
  3. Model-Based Route Agent (A* / Informed Search with Graph Model)

Author: Sepehr Saberi (Coding & Implementation Lead)
Project: Urban Route Finding - AI Lab
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# --- Simulation Hyperparameters ---
GRID_SIZE = 6          # 6x6 urban intersection grid
MAX_STEPS = 60         # Maximum movement budget per run
RUNS = 100             # Number of Monte Carlo experiments per agent
OBSTACLE_PROB = 0.15   # Probability of street closure / severe traffic jam


class UrbanEnvironment:
    """Represents a 2D city street network with obstacles and start/goal points."""
    def __init__(self, size=6, obstacle_prob=0.15, rng=None):
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Build grid graph
        self.graph = nx.grid_2d_graph(size, size)
        
        # Add random road blocks/traffic jams (excluding start and goal)
        self.blocked_nodes = set()
        for x in range(size):
            for y in range(size):
                if (x, y) != self.start and (x, y) != self.goal:
                    if self.rng.random() < obstacle_prob:
                        self.blocked_nodes.add((x, y))
                        
        # Remove blocked nodes from traversable graph
        for node in self.blocked_nodes:
            if self.graph.has_node(node):
                self.graph.remove_node(node)

    def get_valid_neighbors(self, pos):
        """Returns reachable adjacent intersections."""
        if not self.graph.has_node(pos):
            return []
        return list(self.graph.neighbors(pos))

    def manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def simulate_urban_route(agent_type, seed, grid_size=GRID_SIZE, max_steps=MAX_STEPS):
    """
    Simulates a single episode of navigation in the urban grid.
    Returns:
        dict: Performance metrics including score, steps taken, and success flag.
    """
    rng = np.random.default_rng(seed)
    env = UrbanEnvironment(size=grid_size, obstacle_prob=OBSTACLE_PROB, rng=rng)
    
    current_pos = env.start
    goal = env.goal
    steps_taken = 0
    reached_goal = False
    
    # Model-based agent pre-computes optimal route if reachable
    planned_path = []
    if agent_type == "model":
        if nx.has_path(env.graph, env.start, env.goal):
            # A* search with Manhattan heuristic
            planned_path = nx.astar_path(
                env.graph, env.start, env.goal,
                heuristic=lambda a, b: env.manhattan_distance(a, b)
            )
            # Remove start node from planned moves
            planned_path = planned_path[1:]

    # Simulation loop
    for step in range(1, max_steps + 1):
        steps_taken = step
        
        if current_pos == goal:
            reached_goal = True
            break
            
        neighbors = env.get_valid_neighbors(current_pos)
        if not neighbors:
            # Trapped at dead end
            break
            
        # Agent action selection
        if agent_type == "random":
            # Select random adjacent intersection
            next_pos = rng.choice(neighbors)
            next_pos = tuple(next_pos)
            
        elif agent_type == "reflex":
            # Simple reflex / greedy: choose neighbor closest to goal
            best_neighbor = min(
                neighbors,
                key=lambda n: env.manhattan_distance(n, goal) + rng.random() * 0.01
            )
            next_pos = best_neighbor
            
        elif agent_type == "model":
            # Model-based agent follows optimal A* planned path
            if planned_path:
                next_pos = planned_path.pop(0)
            else:
                # If no complete path was found, fall back to best local neighbor
                next_pos = min(neighbors, key=lambda n: env.manhattan_distance(n, goal))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        current_pos = next_pos

    if current_pos == goal:
        reached_goal = True

    # Performance Measure (PEAS):
    # - +100 bonus for reaching goal
    # - -1 movement penalty for every step taken (energy/time consumption)
    # - Penalty proportional to final distance from goal if not reached
    final_dist = env.manhattan_distance(current_pos, goal)
    score = (100 if reached_goal else 0) - steps_taken - (final_dist * 5)

    return {
        "agent": agent_type,
        "score": score,
        "steps": steps_taken,
        "success": 1 if reached_goal else 0,
        "final_distance": final_dist
    }


def run_experiments():
    """Runs 100 trials for each agent and generates statistical summary and plot."""
    print("=" * 65)
    print("Urban Route Finding: Running Monte Carlo Simulation (100 runs)")
    print("=" * 65)
    
    agent_names = ["random", "reflex", "model"]
    records = []

    for agent in agent_names:
        for seed in range(RUNS):
            res = simulate_urban_route(agent, seed)
            records.append(res)

    df = pd.DataFrame(records)

    # Compute summary statistics
    summary = df.groupby("agent").agg({
        "score": ["mean", "std"],
        "steps": ["mean", "std"],
        "success": ["mean"]
    }).round(2)

    print("\nSimulation Results Summary:")
    print(summary)

    # Plot Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot 1: Score with standard deviation
    score_means = df.groupby("agent")["score"].mean()
    score_stds = df.groupby("agent")["score"].std()
    score_means.plot(
        kind="bar", yerr=score_stds, capsize=6, ax=axes[0],
        color=["#94a3b8", "#38bdf8", "#4f46e5"], edgecolor="#1e293b"
    )
    axes[0].set_title("Average Navigation Score (100 runs, 6x6 grid)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Score (Reward - Step Cost)")
    axes[0].set_xlabel("Agent Architecture")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Success Rate
    success_rate = (df.groupby("agent")["success"].mean() * 100)
    success_rate.plot(
        kind="bar", ax=axes[1],
        color=["#f87171", "#fbbf24", "#34d399"], edgecolor="#1e293b"
    )
    axes[1].set_title("Goal Arrival Success Rate (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_xlabel("Agent Architecture")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = "urban_route_comparison.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nPlot successfully saved to: {plot_path}")
    
    return df, summary


if __name__ == "__main__":
    run_experiments()
