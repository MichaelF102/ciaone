#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define G 6.67430e-11

typedef struct {
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
} Body;

double rand_double() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

void init_bodies(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].x = rand_double();
        bodies[i].y = rand_double();
        bodies[i].z = rand_double();
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0;
    }
}

void reset_forces(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].fx = bodies[i].fy = bodies[i].fz = 0;
    }
}

void add_forces(Body *bodies, int n) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double dist = sqrt(dx * dx + dy * dy + dz * dz + 1e-9);
                double force = G / (dist * dist * dist + 1e-9);
                bodies[i].fx += force * dx;
                bodies[i].fy += force * dy;
                bodies[i].fz += force * dz;
            }
        }
    }
}

void update_bodies(Body *bodies, int n, double dt) {
    for (int i = 0; i < n; i++) {
        bodies[i].vx += bodies[i].fx * dt;
        bodies[i].vy += bodies[i].fy * dt;
        bodies[i].vz += bodies[i].fz * dt;
        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

void simulate(int nBodies, int steps, double dt, int parallel) {
    Body *bodies = malloc(sizeof(Body) * nBodies);
    init_bodies(bodies, nBodies);

    clock_t start = clock();

    for (int step = 0; step < steps; step++) {
        reset_forces(bodies, nBodies);
        add_forces(bodies, nBodies);
        update_bodies(bodies, nBodies, dt);
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Simulated %d bodies for %d steps (parallel: %s) in %.3f seconds\n",
           nBodies, steps, parallel ? "true" : "false", elapsed);

    free(bodies);
}
int main() {
    int bodyCounts[] = {100, 200, 500};
    int stepCounts[] = {100, 200};
    double dt = 0.01;

#ifdef _OPENMP
    printf("OpenMP enabled with %d threads.\n", omp_get_max_threads());
#else
    printf("Running without OpenMP.\n");
#endif

    for (int i = 0; i < sizeof(bodyCounts) / sizeof(bodyCounts[0]); i++) {
        for (int j = 0; j < sizeof(stepCounts) / sizeof(stepCounts[0]); j++) {
            int n = bodyCounts[i];
            int steps = stepCounts[j];

            simulate(n, steps, dt, 0); // single-threaded
#ifdef _OPENMP
            simulate(n, steps, dt, 1); // parallel
#endif
            printf("--------------------------------------------------\n");
        }
    }

    return 0;
}

