import java.util.concurrent.*;
import java.lang.management.*;

public class nbody {

    static class Body {
        double x, y, z;
        double vx, vy, vz;
        double fx, fy, fz;

        public Body(double x, double y, double z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public void resetForce() {
            fx = fy = fz = 0;
        }

        public void addForce(Body b) {
            double G = 6.67430e-11;
            double dx = b.x - x;
            double dy = b.y - y;
            double dz = b.z - z;
            double dist = Math.sqrt(dx * dx + dy * dy + dz * dz + 1e-9);
            double force = G / (dist * dist + 1e-9);
            fx += force * dx;
            fy += force * dy;
            fz += force * dz;
        }

        public void update(double dt) {
            vx += fx * dt;
            vy += fy * dt;
            vz += fz * dt;
            x += vx * dt;
            y += vy * dt;
            z += vz * dt;
        }
    }

    // Print current memory stats
    public static void printMemoryUsage() {
        Runtime rt = Runtime.getRuntime();
        long used = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        long total = rt.totalMemory() / (1024 * 1024);
        System.out.printf("Memory Used: %d MB / %d MB%n", used, total);
    }

    // Print current thread stats
    public static void printThreadStats() {
        ThreadMXBean threadMXBean = ManagementFactory.getThreadMXBean();
        int threadCount = threadMXBean.getThreadCount();
        System.out.println("Active Threads: " + threadCount);
    }

    public static void simulate(int numBodies, int numSteps, double dt, boolean parallel) {
        Body[] bodies = new Body[numBodies];
        for (int i = 0; i < numBodies; i++) {
            bodies[i] = new Body(Math.random(), Math.random(), Math.random());
        }

        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        long start = System.nanoTime();

        for (int step = 0; step < numSteps; step++) {
            // Reset forces
            for (Body b : bodies) b.resetForce();

            if (parallel) {
                CountDownLatch latch = new CountDownLatch(numBodies);
                for (int i = 0; i < numBodies; i++) {
                    int finalI = i;
                    executor.submit(() -> {
                        for (int j = 0; j < bodies.length; j++) {
                            if (j != finalI) bodies[finalI].addForce(bodies[j]);
                        }
                        latch.countDown();
                    });
                }
                try {
                    latch.await();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                for (int i = 0; i < numBodies; i++) {
                    for (int j = 0; j < numBodies; j++) {
                        if (i != j) bodies[i].addForce(bodies[j]);
                    }
                }
            }

            for (Body b : bodies) b.update(dt);
        }

        long end = System.nanoTime();
        double seconds = (end - start) / 1e9;

        System.out.printf("Simulated %d bodies for %d steps in %.3f seconds (parallel: %b)%n", numBodies, numSteps, seconds, parallel);
        printMemoryUsage();
        printThreadStats();
        executor.shutdown();
    }

    public static void main(String[] args) {
        // You can tune these parameters
        int[] numBodiesList = {100, 200, 500};
        int[] stepList = {100, 500};
        double dt = 0.01;

        for (int n : numBodiesList) {
            for (int steps : stepList) {
                simulate(n, steps, dt, false);   // single-threaded
                simulate(n, steps, dt, true);    // multi-threaded
                System.out.println("----------------------------------------------------");
            }
        }
    }
}
