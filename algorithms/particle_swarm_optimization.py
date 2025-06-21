"""
population-based stochastic optimization technique inspired by the social behavior of birds and fish:
- A swarm of particles explores the solution space, each with position and velocity.
- This balance between c1 and c2 drives exploration and exploitation without gradient information.
"""
import numpy as np
import time

class ParticleSwarmOptimization:
    """
    objective: object with .bounds (tuple of (low, high)) and .evaluate(x) method.
    n_particles: number of particles in the swarm.
    w: inertia weight controls speed (distance).
    c1: cognitive coefficient (pull towards particle's own personal best).
    c2: social coefficient (pull towards global best (swarm best)).
    max_iter: number of iterations to run.
    """
    def __init__(
        self,
        objective,
        n_particles=30,
        w=0.5,
        c1=1.5,
        c2=1.5,
        max_iter=100
    ):
        self.obj = objective
        self.bounds = objective.bounds  # (low, high)
        self.dim = len(objective.global_min)
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

        # Initialize particle positions and velocities
        low, high = self.bounds
        # Choose random values for positions of the particles in bounds
        self.positions = np.random.uniform(low, high, (n_particles, self.dim))
        # Init all particles velocities in start to zero
        self.velocities = np.zeros((n_particles, self.dim))

        # Initialize personal best positions and their evaluations
        self.pbest_positions = self.positions.copy()
        evals = np.apply_along_axis(self.obj.evaluate, 1, self.positions)
        self.pbest_scores = evals.copy()

        # Initialize global best by finding best value in personal best evals
        best_idx = np.argmin(self.pbest_scores)
        self.gbest_position = self.pbest_positions[best_idx].copy()
        self.gbest_score = self.pbest_scores[best_idx]

        # History tracking
        self.history = []  # best global score per iteration
        self.total_time = 0

    def _update_velocity(self, i):
        """ Update velocity of particle i. """
        """ vᵢ = w·vᵢ + c₁·r₁·(pbestᵢ − xᵢ) + c₂·r₂·(gbest − xᵢ) """
        r1 = np.random.rand(self.dim)
        r2 = np.random.rand(self.dim)
        cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
        social = self.c2 * r2 * (self.gbest_position - self.positions[i])
        # inertia component
        new_v = self.w * self.velocities[i] + cognitive + social
        return new_v

    def _update_position(self, i):
        """Update position of particle i and clip to bounds."""
        """ xᵢ = xᵢ + vᵢ and clipped to bounds """
        new_pos = self.positions[i] + self.velocities[i]
        low, high = self.bounds
        return np.clip(new_pos, low, high)

    def run(self):
        """
        Execute the PSO algorithm.
        Returns:
            gbest_position: best solution found.
            gbest_score: objective value at gbest_position.
        """
        start_time = time.perf_counter()

        for _ in range(self.max_iter):
            for i in range(self.n_particles):
                # Update velocity and position
                self.velocities[i] = self._update_velocity(i)
                self.positions[i] = self._update_position(i)

                # Evaluate new position
                score = self.obj.evaluate(self.positions[i])

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_scores[i] = score

                    # Update global best
                    if score < self.gbest_score:
                        self.gbest_position = self.positions[i].copy()
                        self.gbest_score = score

            # Record global best performance
            self.history.append(self.gbest_score)

        self.total_time = time.perf_counter() - start_time