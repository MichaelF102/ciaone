#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

const double G = 6.67430e-11;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;

    Body() {
        x = rand_range();
        y = rand_range();
        z = rand_range();
        vx = vy = vz = 0;
    }

    static double rand_range() {
        static mt19937_64 rng(42); // Deterministic seed
        static uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(rng);
    }

    void reset_force() {
        fx = fy = fz = 0;
    }

    void add_force(const Body& other) {
        double dx = other.x - x;
        double dy = other.y - y;
        double dz = other.z - z;
        double dist = sqrt(dx * dx + dy * dy + dz * dz + 1e-9);
        double force = G / (dist * dist * dist + 1e-9);
        fx += force * dx;
        fy += force * dy;
        fz += force * dz;
    }

    void update(double dt) {
        vx += fx * dt;
        vy += fy * dt;
        vz += fz * dt;
        x += vx * dt;
        y += vy * dt;
        z += vz * dt;
    }
};

void simulate(int nBodies, int steps, double dt, bool parallel = false) {
    vector<Body> bodies(nBodies);
    auto t_start = high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        // Reset forces
        for (auto& b : bodies)
            b.reset_force();

        // Compute forces
        if (parallel) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int i = 0; i < nBodies; ++i) {
                for (int j = 0; j < nBodies; ++j) {
                    if (i != j)
                        bodies[i].add_force(bodies[j]);
                }
            }
        } else {
            for (int i = 0; i < nBodies; ++i) {
                for (int j = 0; j < nBodies; ++j) {
                    if (i != j)
                        bodies[i].add_force(bodies[j]);
                }
            }
        }

        // Update positions
        for (auto& b : bodies)
            b.update(dt);
    }

    auto t_end = high_resolution_clock::now();
    double elapsed = duration<double>(t_end - t_start).count();

    cout << fixed << setprecision(4);
    cout << "Simulated " << nBodies << " bodies for " << steps << " steps ";
    cout << "(parallel: " << boolalpha << parallel << ") ";
    cout << "in " << elapsed << " seconds.\n";
}

int main() {
    vector<int> bodyCounts = {100, 200, 500};   // Sizes for test
    vector<int> stepCounts = {100, 200};        // Steps per size
    double dt = 0.01;

    cout << "=== C++ N-Body Simulation Profiler ===\n";
#ifdef _OPENMP
    cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    cout << "Compiled without OpenMP support.\n";
#endif
    cout << "----------------------------------------\n";

    for (int n : bodyCounts) {
        for (int steps : stepCounts) {
            simulate(n, steps, dt, false);  // Single-threaded
#ifdef _OPENMP
            simulate(n, steps, dt, true);   // Multi-threaded
#endif
            cout << "----------------------------------------\n";
        }
    }

    return 0;
}
