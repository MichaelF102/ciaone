import time
import random
import math
import cProfile
import pstats
from memory_profiler import memory_usage

G = 6.67430e-11  # gravitational constant

class Body:
    def __init__(self):
        self.x = random.uniform(-1, 1)
        self.y = random.uniform(-1, 1)
        self.z = random.uniform(-1, 1)
        self.vx = self.vy = self.vz = 0
        self.fx = self.fy = self.fz = 0

    def reset_force(self):
        self.fx = self.fy = self.fz = 0

    def add_force(self, other):
        dx = other.x - self.x
        dy = other.y - self.y
        dz = other.z - self.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz + 1e-9)
        force = G / (dist**3 + 1e-9)
        self.fx += force * dx
        self.fy += force * dy
        self.fz += force * dz

    def update(self, dt):
        self.vx += self.fx * dt
        self.vy += self.fy * dt
        self.vz += self.fz * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt


def simulate(n_bodies=100, n_steps=100, dt=0.01):
    bodies = [Body() for _ in range(n_bodies)]

    for step in range(n_steps):
        for body in bodies:
            body.reset_force()
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                bodies[i].add_force(bodies[j])
                bodies[j].add_force(bodies[i])
        for body in bodies:
            body.update(dt)

def profile_simulation(n_bodies, n_steps):
    print(f"\n=== Profiling: {n_bodies} bodies, {n_steps} steps ===")
    t0 = time.time()
    mem_usage = memory_usage((simulate, (n_bodies, n_steps)), interval=0.1)
    elapsed = time.time() - t0
    print(f"Execution Time: {elapsed:.3f} s")
    print(f"Peak Memory Usage: {max(mem_usage):.2f} MiB")


def run_with_cprofile(n_bodies, n_steps):
    print(f"\n=== cProfile: {n_bodies} bodies, {n_steps} steps ===")
    profiler = cProfile.Profile()
    profiler.enable()
    simulate(n_bodies, n_steps)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumtime').print_stats(10)

def main():
    test_cases = [
        (50, 100),
        (100, 100),
        (200, 50),
        (300, 20)
    ]

    for n, steps in test_cases:
        profile_simulation(n, steps)
        run_with_cprofile(n, steps)


if __name__ == "__main__":
    main()
