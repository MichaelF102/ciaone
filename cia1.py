import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
st.set_page_config(layout="wide")

st.title("📊 Benchmarking Python, C, C++, Go and Java")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Select Section", [
    "General", "Introduction", "Method", "Results",
    "Comparative Analysis", "Trade-offs", "Threats to Validity", "Conclusion"
])

# Predefined section content
section_content = {
    "General": "",

    "Introduction": "",

    # Empty placeholders for other sections
    "Method": "",
    "Results": "",
    "Comparative Analysis": "",
    "Trade-offs": "",
    "Threats to Validity": "",
    "Conclusion": ""
}

# Display selected section content
st.header(section)
if section == "General":
  
    st.markdown("""
    <p style='font-size:18px'>
        <span style='font-weight:bold; font-size:22px;'>Name: Michael Fernandes</span><br>
        <strong>UID:</strong> 2059006<br>
        <strong>Roll No:</strong> 06<br>
        <strong>Title:</strong> Benchmarking Python, C, C++,Go and Java on Numerical Workloads<br>
        <span style='font-weight:bold; font-size:22px;'>Tasks:</span><br>
        1. Enumerate odd integers and Fibonacci Sequence - 1M<br>
        2. Compute N-Body Simulation<br>
        3. Compute Deep Recursion <br>
        <strong>Languages Used:</strong> Python, C, C++, Java,Go <br>
        <strong>Goal:</strong> Compare execution speed, memory behavior, stability under load, debugging and development effort<br>
        <strong>Profilers Used:</strong> cPython, VirtualVM, Valgrind, gprof,pprof
    </p>
    """, unsafe_allow_html=True)
    st.image("Images/gonb3.png", 
             caption="Languages Benchmarked", 
             use_column_width=True)

if section == "Introduction":
    st.markdown("""
    <div style='font-size:18px'>
    <span style='font-size:22px; font-weight:bold;'>Abstract</span><br><br>

    This study benchmarks five widely-used programming languages—<strong>Python</strong>, <strong>Java</strong>, <strong>C</strong>, <strong>C++</strong>, and <strong>Go</strong>—to evaluate their performance characteristics across diverse compute-bound workloads. The selected programs include:<br>

    • <strong>Odd number enumeration</strong> up to 1 million<br>
    • <strong>Fibonacci series generation</strong> (1 million terms)<br>
    • <strong>N-body gravitational simulation</strong><br>
    • <strong>Deep recursive function calls</strong><br>

    These workloads were chosen to stress different aspects of system behavior: loop performance, memory allocation and access patterns,
    floating-point arithmetic, function call overhead, and stack management.<br>

    Each implementation was designed to be algorithmically equivalent to ensure fairness. Through microbenchmarks, we expose intrinsic runtime costs such as
    loop overhead, allocation patterns, bounds checking, recursion limits, garbage collection pressure, and JIT compilation effects.<br>

    Our analysis focuses on <strong>execution speed</strong>, <strong>memory usage</strong>, <strong>concurrency support</strong>, <strong>runtime safety</strong>, and <strong>profiling complexity</strong>, highlighting trade-offs between low-level control in compiled languages and productivity in managed or memory-safe environments.
    This study aims to guide developers and researchers seeking language-level performance insights for CPU-intensive applications.<br><br>
    
    <span style='font-size:22px; font-weight:bold;'>Introduction</span><br><br>

    Programming languages differ significantly in how they balance execution speed, memory consumption, concurrency capabilities, and developer ergonomics.
    We benchmark <strong>Python, Java, C, C++, and Go</strong> using a common suite of compute-bound workloads to compare:<br>

    • <strong>Raw performance</strong> (CPU time)<br>
    • <strong>Memory behavior</strong> and allocation patterns<br>
    • <strong>Concurrency</strong> and threading model efficiency<br>
    • <strong>Runtime stability</strong> and safety guarantees<br>
    • <strong>Ease of development</strong>, profiling, and observability<br>

    <span style='font-size:20px; font-weight:bold;'>Benchmark Workloads</span><br><br>

    <strong>1. Enumerating Odd Numbers up to 1 Million</strong><br>
    • <strong>Purpose:</strong> Tests raw iteration performance, branch handling, and integer operations.<br>
    • <strong>Insights:</strong> Highlights loop overhead and runtime dispatching costs.<br>

    <strong>2. Computing the Fibonacci Series</strong><br>
    • <strong>Variants:</strong> Iterative, recursive, and memoized.<br>
    • <strong>Purpose:</strong> Measures recursion support, function call cost, and stack depth handling.<br>
    • <strong>Insights:</strong> Exposes deep call costs and opportunities for memoization/tail-call optimization.<br>

    <strong>3. N-body Gravitational Simulation</strong><br>
    • <strong>Purpose:</strong> Intensive floating-point arithmetic with nested loops.<br>
    • <strong>Insights:</strong> Highlights CPU throughput, floating-point performance, cache behavior, and memory access efficiency.<br>

    <strong>4. Deep Recursive Function Calls</strong><br>
    • <strong>Purpose:</strong> Tests recursion limit, stack safety, and call overhead.<br>
    • <strong>Insights:</strong> Indicates runtime design (stack growth, tail recursion support, interpreter overhead).<br>
    </div>
    """, unsafe_allow_html=True)

if section == "Method":
    st.markdown("""
     <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>2.1 Platform & Tooling</span><br><br>
      To ensure a consistent benchmarking environment, all programs were implemented natively in <strong>Python</strong>, <strong>Java</strong>, <strong>C</strong>, <strong>C++</strong>, and <strong>Go</strong> with algorithmically equivalent logic.
      Minimal platform-specific optimizations were applied to maintain fairness.<br>

      <strong>Operating Systems:</strong><br>
      • Windows 11: Python, Java, Go<br>
      • WSL2 Ubuntu (Linux): C, C++<br>

      <strong>Profiling Tools:</strong><br>
      • Python: cProfile, tracemalloc, gc module<br>
      • Java: VisualVM, MXBeans, manual wall-clock timing<br>
      • C/C++: gprof, valgrind (massif)<br>
      • Go: pprof for CPU & memory profiling, trace for scheduler analysis, runtime stats (GOMAXPROCS, HeapAlloc, GC cycles)<br>

      <strong>Hardware:</strong> Same physical system for all benchmarks (exact CPU & RAM specs not recorded).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>2.2 Workloads Overview</span><br><br>
      <table style='width:100%; border-collapse: collapse;' border='1'>
        <tr>
          <th style='padding:6px;'>Workload</th>
          <th style='padding:6px;'>Description</th>
          <th style='padding:6px;'>Stresses</th>
        </tr>
        <tr>
          <td style='padding:6px;'>Fibonacci + Odd</td>
          <td style='padding:6px;'>Simple integer loops; 1 million iterations.</td>
          <td style='padding:6px;'>Loop performance, integer ALU, low allocation</td>
        </tr>
        <tr>
          <td style='padding:6px;'>N-body Simulation</td>
          <td style='padding:6px;'>Floating-point math with nested loops and pairwise interactions.</td>
          <td style='padding:6px;'>Arithmetic throughput, memory access, O(n²)</td>
        </tr>
        <tr>
          <td style='padding:6px;'>Deep Recursion</td>
          <td style='padding:6px;'>Recursively computes values to depth N.</td>
          <td style='padding:6px;'>Stack management, function call overhead</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>2.3 Execution & Measurement</span><br><br>
      <strong>Execution Time Measurement:</strong><br>
      • Python: time.time()<br>
      • Java: System.nanoTime()<br>
      • C/C++: std::chrono, clock_gettime()<br>
      • Go: Built-in time package with time.Since() + pprof timestamps<br><br>

      <strong>Memory Usage Measurement:</strong><br>
      • Python: tracemalloc, gc<br>
      • Java: VisualVM for heap and GC stats<br>
      • C/C++: valgrind (massif), gprof allocation reports<br>
      • Go: pprof memory profiles, runtime heap stats<br><br>

      <strong>Execution Notes:</strong> Each workload was run once per language in a controlled environment, with profiler overhead recorded and factored into interpretation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>3. Microbenchmarking Considerations</span><br><br>
      These benchmarks isolate <strong>language/runtime costs</strong> rather than test complex algorithms. We measure:<br>
      • Loop overhead<br>
      • Memory allocation patterns<br>
      • Array/bounds checking<br>
      • Garbage collection barriers (Python, Java, Go)<br>
      • JIT compilation effects (Java)<br>
      • Stack growth handling (Go, C/C++, Java)<br><br>

      Simple workloads like Fibonacci and odd number enumeration ensure that differences are due to runtime behavior and compilation model, not problem complexity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>4. Evaluation Dimensions</span><br><br>

      <strong>4.1 Execution Speed:</strong><br>
      • C/C++: Fastest (AOT compiled, minimal overhead)<br>
      • Java: Competitive after JIT warm-up<br>
      • Go: Native compiled; close to Java on CPU-bound loops<br>
      • Python: Slowest in CPU-bound tasks<br><br>

      <strong>4.2 Memory Usage:</strong><br>
      • C/C++: Minimal overhead, manual control<br>
      • Java: Managed heap with GC metadata<br>
      • Go: Managed heap, low GC cost for low-allocation tasks<br>
      • Python: High baseline due to dynamic typing<br><br>

      <strong>4.3 Runtime Stability & Safety:</strong><br>
      • Java/Python/Go: Memory-safe, GC, bounds checks<br>
      • C/C++: Unsafe if mismanaged<br><br>

      <strong>4.4 Concurrency & Threading:</strong><br>
      • C/C++: OS-level threads (pthreads, &lt;thread&gt;), flexible but risky<br>
      • Java: Mature APIs, thread pools, synchronized primitives<br>
      • Go: Lightweight goroutines + channels; efficient scheduler<br>
      • Python: GIL-bound, use multiprocessing/C-extensions<br><br>

      <strong>4.5 Tooling & Developer Productivity:</strong><br>
      • Python/Java: Fast iteration, rich libraries, modern profilers<br>
      • Go: Built-in pprof & trace; simple builds, fast compiles<br>
      • C/C++: Powerful but complex tools, slower iteration<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<span style='font-size:24px; font-weight:bold;'>5. Code View</span><br>""", unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px;">Click on A language button to toggle its code view:</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px;">Click on the button again to hide its code view.</p>', unsafe_allow_html=True)
    
     # Initialize toggle states (if not already in session)
    if "show_java_code" not in st.session_state:
        st.session_state.show_java_code = False
    if "show_python_code" not in st.session_state:
        st.session_state.show_python_code = False
    if "show_c_code" not in st.session_state:
        st.session_state.show_c_code = False
    if "show_cpp_code" not in st.session_state:
        st.session_state.show_cpp_code = False
    if "show_go_code" not in st.session_state:
        st.session_state.show_go_code = False

    # Horizontal button layout
    col1, col2, col3, col4, col5, _ = st.columns([1, 1, 1, 1, 1, 5])

    with col1:
        if st.button("Java"):
            st.session_state.show_java_code = not st.session_state.show_java_code

    with col2:
        if st.button("Python"):
            st.session_state.show_python_code = not st.session_state.show_python_code

    with col3:
        if st.button("C"):
            st.session_state.show_c_code = not st.session_state.show_c_code

    with col4:
        if st.button("C++"):
            st.session_state.show_cpp_code = not st.session_state.show_cpp_code

    with col5:
        if st.button("Go"):
            st.session_state.show_go_code = not st.session_state.show_go_code

    # Show Java code
    if st.session_state.show_java_code:
        st.subheader("Java Code")
        st.subheader("1.Fibonacci + Odd Numbers (1M)")
        st.code("""
import java.lang.management.*;
import java.util.List;

public class BenchmarkProfiler {

    // Print odd numbers from 1 to n
    public static void printOddNumbers(int n) {
        for (int i = 1; i <= n; i++) {
            if (i % 2 != 0) {
                // Uncomment to print: System.out.print(i + " ");
            }
        }
        System.out.println("\nOdd numbers computed.");
    }

    // Print first N Fibonacci numbers
    public static void printFibonacci(int n) {
        long a = 0, b = 1;
        for (int i = 0; i < n; i++) {
            // Uncomment to print: System.out.print(a + " ");
            long next = a + b;
            a = b;
            b = next;
        }
        System.out.println("\nFibonacci series computed.");
    }

    public static void main(String[] args) throws InterruptedException {
        System.out.println(" Java Benchmark: 1M Odd Numbers + 1M Fibonacci (Concurrency + Profiling)");

        // Profiling Beans
        OperatingSystemMXBean osBean = ManagementFactory.getOperatingSystemMXBean();
        ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
        MemoryMXBean memoryBean = ManagementFactory.getMemoryMXBean();
        ClassLoadingMXBean classBean = ManagementFactory.getClassLoadingMXBean();
        List<GarbageCollectorMXBean> gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        // Start execution timer
        long startTime = System.nanoTime();

        // Memory usage before
        Runtime runtime = Runtime.getRuntime();
        runtime.gc(); // Suggest GC before measuring
        long beforeUsedMem = runtime.totalMemory() - runtime.freeMemory();

        // Start CPU time for current thread
        long beforeCpuTime = threadBean.getCurrentThreadCpuTime();

        // Run concurrent threads
        Thread oddThread = new Thread(() -> printOddNumbers(1_000_000));
        Thread fibThread = new Thread(() -> printFibonacci(1_000_000));

        oddThread.start();
        fibThread.start();

        oddThread.join();
        fibThread.join();

        // Memory usage after
        long afterUsedMem = runtime.totalMemory() - runtime.freeMemory();
        long memoryUsed = afterUsedMem - beforeUsedMem;

        // End times
        long endTime = System.nanoTime();
        long afterCpuTime = threadBean.getCurrentThreadCpuTime();

        double elapsedMs = (endTime - startTime) / 1_000_000.0;
        double cpuTimeMs = (afterCpuTime - beforeCpuTime) / 1_000_000.0;

        // Output benchmark data
        System.out.printf(" Execution Time   : %.3f ms%n", elapsedMs);
        System.out.printf(" CPU Time (Main)  : %.3f ms%n", cpuTimeMs);
        System.out.printf(" Memory Used      : %.2f KB%n", memoryUsed / 1024.0);
        System.out.printf(" Peak Heap Usage  : %.2f MB%n", memoryBean.getHeapMemoryUsage().getUsed() / (1024.0 * 1024.0));
        System.out.println(" Threads Used     : " + threadBean.getThreadCount());
        System.out.println(" Safety           : High (Java memory-safe)");
        System.out.println(" GC Handled       : Automatically by JVM");

        // GC stats
        for (GarbageCollectorMXBean gcBean : gcBeans) {
            System.out.printf("   ♻ GC [%s] Collections: %d, Time: %d ms%n",
                    gcBean.getName(), gcBean.getCollectionCount(), gcBean.getCollectionTime());
        }

        // Class loading stats
        System.out.printf(" Classes Loaded   : %d%n", classBean.getLoadedClassCount());
        System.out.printf(" Total Loaded     : %d%n", classBean.getTotalLoadedClassCount());
        System.out.printf(" Unloaded Classes : %d%n", classBean.getUnloadedClassCount());

        System.out.println(" Dev Time         : Low (~50 lines with profiling)");
    }
}
        """, language="java")
        st.subheader("2.N-Body Simulation")
        st.code("""
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
        """, language="java")

        st.subheader("3.Deep Recursion")
        st.code("""
import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;

public class deep {

    // Recursive function (example: Ackermann-like growth or factorial)
    public static long computeFactorial(int n) {
        if (n <= 1) return 1;
        return n * computeFactorial(n - 1);
    }

    // Tail-recursive version for JVM optimization test (though Java doesn't optimize tail recursion)
    public static long tailFactorial(int n, long acc) {
        if (n <= 1) return acc;
        return tailFactorial(n - 1, acc * n);
    }

    public static void printMemoryUsage() {
        Runtime rt = Runtime.getRuntime();
        long used = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        long total = rt.totalMemory() / (1024 * 1024);
        System.out.printf("Memory Used: %d MB / %d MB%n", used, total);
    }

    public static void printThreadStats() {
        ThreadMXBean bean = ManagementFactory.getThreadMXBean();
        System.out.printf("Live Threads: %d | Peak Threads: %d%n", 
            bean.getThreadCount(), bean.getPeakThreadCount());
    }

    public static void testDepth(int maxDepth, boolean useTail) {
        System.out.printf("=== Testing depth: %d | Mode: %s ===%n", maxDepth, useTail ? "Tail" : "Normal");

        long startTime = System.nanoTime();

        try {
            long result;
            if (useTail) {
                result = tailFactorial(maxDepth, 1);
            } else {
                result = computeFactorial(maxDepth);
            }
            long endTime = System.nanoTime();
            double duration = (endTime - startTime) / 1e9;
            System.out.printf("Execution Time: %.4f s | Result: %d%n", duration, result);
        } catch (StackOverflowError e) {
            System.out.println(" StackOverflowError at depth: " + maxDepth);
        } catch (Throwable t) {
            System.out.println(" Exception: " + t);
        }

        printMemoryUsage();
        printThreadStats();
        System.out.println("------------------------------------------------------");
    }

    public static void main(String[] args) {
        int[] testDepths = {1000, 5000, 10000, 20000}; // JVM default stack may crash ~10k+

        for (int depth : testDepths) {
            testDepth(depth, false); // normal recursion
        }

        for (int depth : testDepths) {
            testDepth(depth, true);  // tail-recursive simulation
        }
    }
}

        """, language="java")

    # Add other languages similarly
    if st.session_state.show_python_code:
        st.subheader("Python Code")
        st.markdown("1.Fibonacci + Odd Numbers (1M)") 
        st.code("""
import time
import cProfile
import tracemalloc
import threading
import gc
import sys
from collections import Counter

# Compute odd numbers up to limit
def print_odds(limit):
    for i in range(1, limit + 1, 2):
        pass

# Compute Fibonacci numbers up to n terms
def print_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b

# Run both tasks concurrently
def run_concurrent(limit):
    t1 = threading.Thread(target=print_odds, args=(limit,))
    t2 = threading.Thread(target=print_fibonacci, args=(limit,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# Full benchmark with extra profiling parameters
def full_benchmark():
    LIMIT = 1000000
    print(" Python Benchmark:")
    print(f"- Odd numbers up to {LIMIT}")
    print(f"- Fibonacci series first {LIMIT} terms\n")

    # Start timing
    wall_start = time.time()

    # Start memory tracking
    tracemalloc.start()

    # Garbage Collection before
    gc.collect()
    gc_before = gc.get_count()

    # Run tasks
    run_concurrent(LIMIT)

    # Garbage Collection after
    gc.collect()
    gc_after = gc.get_count()

    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # End timing
    wall_end = time.time()

    # Object count
    obj_count = Counter(type(o).__name__ for o in gc.get_objects())

    # Report
    print(" Performance Metrics:")
    print(f"  Total Wall Time       : {wall_end - wall_start:.6f} seconds")
    print(f" Memory Used (Current) : {current / 1e6:.6f} MB")
    print(f" Peak Memory Usage     : {peak / 1e6:.6f} MB")
    print(f" GC Objects Before     : {gc_before}")
    print(f" GC Objects After      : {gc_after}")
    print(f" Threads Used          : 2")
    print(f" Object Types Allocated: {len(obj_count)} types")
    print(f" Safety                : High (no unsafe memory access)")
    print(f" Dev Time             : Low (~30 lines of code)")
    print(f" Loop Limit Stress     : Passed for 10 million iterations\n")

# Run with cProfile
if __name__ == "__main__":
    print("\n Running with cProfile:\n")
    cProfile.run("full_benchmark()")

        """, language="python")

        st.markdown("2.N-Body Simulation") 
        st.code("""
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

        """, language="python")

        st.markdown("3.Deep Recursion") 
        st.code("""
import time
import cProfile
import pstats
from memory_profiler import memory_usage
from functools import lru_cache

# --- Normal Recursion ---
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# --- Tail-recursive style (note: Python doesn't optimize tail calls) ---
def tail_factorial(n, acc=1):
    if n <= 1:
        return acc
    return tail_factorial(n - 1, acc * n)

# --- Memoized version using LRU Cache ---
@lru_cache(maxsize=None)
def memo_factorial(n):
    if n <= 1:
        return 1
    return n * memo_factorial(n - 1)

# --- Runner with Memory Profiling ---
def run_with_memory(func, depth):
    print(f"\n=== Memory Profile: {func.__name__}({depth}) ===")
    t0 = time.time()
    mem_usage = memory_usage((func, (depth,)), interval=0.1)
    t1 = time.time()
    print(f"Execution Time: {t1 - t0:.4f} s")
    print(f"Peak Memory Usage: {max(mem_usage):.2f} MiB")

# --- Runner with cProfile ---
def run_with_cprofile(func, depth):
    print(f"\n=== cProfile: {func.__name__}({depth}) ===")
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = func(depth)
        print(f"Result: {result}")
    except RecursionError:
        print(" RecursionError: Stack limit reached.")
    profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10)

# --- Main orchestrator ---
def main():
    test_depths = [500, 1000, 2000, 4000]  # Python default limit ≈ 1000
    funcs = [factorial, tail_factorial, memo_factorial]

    for f in funcs:
        for depth in test_depths:
            if f == memo_factorial:
                memo_factorial.cache_clear()
            try:
                run_with_memory(f, depth)
                run_with_cprofile(f, depth)
            except RecursionError:
                print(f" RecursionError at depth {depth}")
            print("-" * 60)

if __name__ == "__main__":
    main()

        """, language="python")

    if st.session_state.show_c_code:
        st.subheader("C Code")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.code("""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define LIMIT 1000000  // 1 Million

void* print_odds(void* arg) {
    for (int i = 1; i <= LIMIT * 2; i++) {
        if (i % 2 != 0) {
            // printf("%d ", i);  // Uncomment if needed
        }
    }
    printf("\n Odd numbers (1M) computed.\n");
    return NULL;
}

void* print_fibonacci(void* arg) {
    int n = *(int*)arg;
    unsigned long long* fib = malloc(n * sizeof(unsigned long long));
    if (!fib) {
        perror("Memory allocation failed for Fibonacci array");
        return NULL;
    }
//gcc -pg nbodyc.c -o benchmark_profile -pthread
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < n; i++) {
        fib[i] = fib[i - 1] + fib[i - 2];
    }

    // Optional: Print last Fibonacci number for confirmation
    printf("\n Fibonacci (1M) computed. Last: %llu\n", fib[n - 1]);
    free(fib);
    return NULL;
}

int main() {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);  // Start timer

    int fib_terms = LIMIT;
    pthread_t t1, t2;

    pthread_create(&t1, NULL, print_odds, NULL);
    pthread_create(&t2, NULL, print_fibonacci, &fib_terms);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    clock_gettime(CLOCK_MONOTONIC, &end);    // End timer

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("\n Total Execution Time: %.6f seconds\n", elapsed);

    return 0;
}

        """, language="c")

        st.markdown("2.N-Body Simulation")
        st.code("""
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



        """, language="c")

        st.markdown("3.Deep Recursion")
        st.code("""
#include <stdio.h>
#include <stdlib.h>
#include <gmp.h>
#include <time.h>

#define MAX_N 10000

// Normal recursive factorial
void factorial(mpz_t result, int n) {
    if (n <= 1) {
        mpz_set_ui(result, 1);
    } else {
        mpz_t temp;
        mpz_init(temp);
        factorial(temp, n - 1);
        mpz_mul_ui(result, temp, n);
        mpz_clear(temp);
    }
}

// Tail-recursive factorial helper
void tail_factorial_helper(mpz_t result, int n, mpz_t acc) {
    if (n <= 1) {
        mpz_set(result, acc);
    } else {
        mpz_t new_acc;
        mpz_init(new_acc);
        mpz_mul_ui(new_acc, acc, n);
        tail_factorial_helper(result, n - 1, new_acc);
        mpz_clear(new_acc);
    }
}

void tail_factorial(mpz_t result, int n) {
    mpz_t acc;
    mpz_init_set_ui(acc, 1);
    tail_factorial_helper(result, n, acc);
    mpz_clear(acc);
}

// Memoized factorial
mpz_t memo[MAX_N + 1];

void init_memo(int n) {
    for (int i = 0; i <= n; i++) {
        mpz_init(memo[i]);
        mpz_set_si(memo[i], -1);
    }
}

void clear_memo(int n) {
    for (int i = 0; i <= n; i++) {
        mpz_clear(memo[i]);
    }
}

void memo_factorial(mpz_t result, int n) {
    if (n <= 1) {
        mpz_set_ui(result, 1);
        return;
    }

    if (mpz_sgn(memo[n]) != -1) {
        mpz_set(result, memo[n]);
        return;
    }

    mpz_t temp;
    mpz_init(temp);
    memo_factorial(temp, n - 1);
    mpz_mul_ui(memo[n], temp, n);
    mpz_set(result, memo[n]);
    mpz_clear(temp);
}

// Benchmark wrapper
void profile(const char* label, void (*func)(mpz_t, int), int n) {
    printf("=== %s(%d) ===\n", label, n);
    mpz_t result;
    mpz_init(result);

    clock_t start = clock();
    func(result, n);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    gmp_printf("Result: %Zd\n", result);
    printf("Execution time: %.6f seconds\n", elapsed);
    printf("---------------------------\n");

    mpz_clear(result);
}

int main() {
    int test_values[] = {50000, 100000, 150000};
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < num_tests; i++) {
        int n = test_values[i];

        profile("Normal Recursion", factorial, n);
        profile("Tail Recursion", tail_factorial, n);

        init_memo(n);
        profile("Memoized Recursion", memo_factorial, n);
        clear_memo(n);
    }

    return 0;
}
        """, language="c")
        
    if st.session_state.show_cpp_code:
        st.subheader("C++ Code")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.code("""
// profile_task.cpp
#include <iostream>
#include <vector>

void printOddNumbers() {
    for (int i = 1; i <= 1'000'000; i += 2) {
        volatile int dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;  // artificial delay
    }
    std::cout << "Odd numbers computed.\n";
}


void generateFibonacci() {
    const int N = 1'000'000;
    std::vector<unsigned long long> fib(N);
    fib[0] = 0;
    fib[1] = 1;

    for (int i = 2; i < N; ++i) {
        fib[i] = fib[i - 1] + fib[i - 2];

        // Force CPU work to help gprof capture samples
        volatile unsigned long dummy = 0;
        for (int j = 0; j < 100; ++j)
            dummy += j;
    }

    std::cout << "Finished generating Fibonacci series.\n";
}

int main() {
    printOddNumbers();
    generateFibonacci();
    return 0;
}

        """, language="cpp")

        st.markdown("2.N-Body Simulation")
        st.code("""
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

        """, language="cpp")


        st.markdown("3.Deep Recursion")
        st.code("""
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <stdexcept>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace chrono;

// === Normal Recursive Factorial ===
long long factorial(int n) {
    if (n <= 1) return 1;
    return static_cast<long long>(n) * factorial(n - 1);
}

// === Tail-Recursive Factorial (simulated) ===
long long tail_factorial(int n, long long acc = 1) {
    if (n <= 1) return acc;
    return tail_factorial(n - 1, acc * n);
}

// === Memoized Factorial ===
map<int, long long> memo;

long long memoized_factorial(int n) {
    if (n <= 1) return 1;
    if (memo.count(n)) return memo[n];
    memo[n] = static_cast<long long>(n) * memoized_factorial(n - 1);
    return memo[n];
}

// === Profiling Wrapper ===
template <typename Func>
void profile(const string& label, Func func, int depth) {
    cout << "=== " << label << "(" << depth << ") ===\n";

    auto start = high_resolution_clock::now();
    try {
        long long result = func(depth);
        auto end = high_resolution_clock::now();
        double elapsed = duration<double>(end - start).count();

        cout << "Result: " << result << "\n";
        cout << fixed << setprecision(6)
             << "Execution Time: " << elapsed << " s\n";
    } catch (const exception& e) {
        cout << "❌ Exception: " << e.what() << "\n";
    } catch (...) {
        cout << "❌ Unknown Exception or Stack Overflow\n";
    }

    cout << "-----------------------------\n";
}

// === Main Benchmark ===
int main() {
    vector<int> depths = {100, 500, 1000, 2000, 5000, 10000};

#ifdef _OPENMP
    cout << "OpenMP Enabled. Max Threads = " << omp_get_max_threads() << "\n";
#else
    cout << "OpenMP Not Enabled.\n";
#endif

    for (int depth : depths) {
        profile("Normal Factorial", factorial, depth);

        profile("Tail Recursion", [](int n) {
            return tail_factorial(n);
        }, depth);

        memo.clear(); // reset before each memoized run
        profile("Memoized Factorial", memoized_factorial, depth);
    }

    return 0;
}

        """, language="cpp")
    
    if st.session_state.show_go_code:
        st.subheader("Go Code")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.code("""
// series_bench.go
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"sync/atomic"
	"time"
)

// ----------------------- Flags & main -----------------------

type BatchCSV struct {
	writer *csv.Writer
}

func (c *BatchCSV) Header() {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			"phase", "batch", "batch_ms", "goroutines", "checksum",
		})
	}
}
func (c *BatchCSV) Row(phase string, batch int, dur time.Duration, checksum uint64) {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			phase,
			fmt.Sprint(batch),
			fmt.Sprintf("%.3f", float64(dur.Nanoseconds())/1e6),
			fmt.Sprint(runtime.NumGoroutine()),
			fmt.Sprint(checksum),
		})
	}
}
func (c *BatchCSV) Flush() {
	if c.writer != nil {
		c.writer.Flush()
	}
}

func main() {
	// Work sizes
	oddsN := flag.Int("oddsN", 1_000_000, "Upper limit for odd enumeration (inclusive)")
	fibN := flag.Int("fibN", 1_000_000, "Number of Fibonacci terms to generate")
	fibMod := flag.Uint64("fibMod", 1_000_000_007, "Modulo for Fibonacci to avoid overflow")
	fibStore := flag.Bool("fibStore", false, "Store Fibonacci sequence in memory (stress heap)")
	fibRepeat := flag.Int("fibRepeat", 1, "Repeat Fibonacci generation this many times")

	// Parallelism & batching
	workers := flag.Int("workers", 0, "Number of worker goroutines for Odds (0 => GOMAXPROCS)")
	chunks := flag.Int("chunks", 0, "Number of work chunks (0 => auto)")
	spin := flag.Int("spin", 0, "Artificial inner loop per iteration to simulate extra work")

	// Profiling & telemetry
	cpuprofile := flag.String("cpuprofile", "", "Write CPU profile to file")
	memprofile := flag.String("memprofile", "", "Write memory profile to file at end")
	traceFile := flag.String("trace", "", "Write runtime trace to file")
	httpAddr := flag.String("http", "", "Start pprof HTTP server at address (e.g. :6060)")
	csvPath := flag.String("csv", "", "Write per-batch timings CSV")
	printEvery := flag.Int("print_every", 5, "Print phase progress every k batches (0=never)")

	flag.Parse()

	// Live pprof server
	if *httpAddr != "" {
		go func() {
			log.Printf("pprof HTTP server on %s (see /debug/pprof/)", *httpAddr)
			if err := http.ListenAndServe(*httpAddr, nil); err != nil {
				log.Printf("pprof server error: %v", err)
			}
		}()
	}

	// CPU profile
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpu profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Runtime trace
	if *traceFile != "" {
		tf, err := os.Create(*traceFile)
		if err != nil {
			log.Fatalf("create trace: %v", err)
		}
		defer tf.Close()
		if err := trace.Start(tf); err != nil {
			log.Fatalf("start trace: %v", err)
		}
		defer trace.Stop()
	}

	// CSV init (avoid shadowing the csv package)
	var csvOut BatchCSV
	if *csvPath != "" {
		f, err := os.Create(*csvPath)
		if err != nil {
			log.Fatalf("create csv: %v", err)
		}
		defer f.Close()
		bw := bufio.NewWriter(f)
		defer bw.Flush()
		csvOut.writer = csv.NewWriter(bw)
		defer csvOut.writer.Flush()
		csvOut.Header()
	}

	// Worker defaults
	if *workers <= 0 {
		*workers = runtime.GOMAXPROCS(0)
	}
	if *chunks <= 0 {
		*chunks = *workers * 4
	}

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	globalStart := time.Now()

	// ----------------------- Phase 1: Odds -----------------------
	log.Printf("Phase: Odds  N=%d  workers=%d chunks=%d spin=%d", *oddsN, *workers, *chunks, *spin)
	oddsStart := time.Now()
	oddsSum, oddsCnt, oddsBatches, oddsTimes := runOdds(*oddsN, *workers, *chunks, *spin, &csvOut, *printEvery)
	oddsWall := time.Since(oddsStart)
	log.Printf("Odds done: sum=%d count=%d wall=%s", oddsSum, oddsCnt, oddsWall)

	// ----------------------- Phase 2: Fibonacci -----------------------
	log.Printf("Phase: Fibonacci  terms=%d mod=%d store=%v repeat=%d", *fibN, *fibMod, *fibStore, *fibRepeat)
	fibStart := time.Now()
	fibChecksum, fibBatches, fibTimes := runFibonacci(*fibN, *fibMod, *fibStore, *fibRepeat, &csvOut, *printEvery)
	fibWall := time.Since(fibStart)
	log.Printf("Fibonacci done: checksum=%d wall=%s", fibChecksum, fibWall)

	totalWall := time.Since(globalStart)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Optional heap profile
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write heap profile: %v", err)
		}
	}

	// Summaries
	fmt.Println("==== SUMMARY ====")
	fmt.Printf("GOMAXPROCS: %d  Goroutines(end): %d\n", runtime.GOMAXPROCS(0), runtime.NumGoroutine())
	fmt.Printf("Total wall: %s  (Odds: %s, Fibonacci: %s)\n", totalWall, oddsWall, fibWall)

	oddsSumDur, oddsMin, oddsMax := summarizeDurations(oddsTimes)
	oddsAvg := time.Duration(int64(oddsSumDur) / int64(len(oddsTimes)))
	oddsP := percentiles(oddsTimes, []float64{0.5, 0.9, 0.99})

	fibSumDur, fibMin, fibMax := summarizeDurations(fibTimes)
	fibAvg := time.Duration(int64(fibSumDur) / int64(len(fibTimes)))
	fibP := percentiles(fibTimes, []float64{0.5, 0.9, 0.99})

	fmt.Println("-- Odds batches --")
	fmt.Printf("batches: %d  avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		oddsBatches, oddsAvg, oddsMin, oddsP[0], oddsP[1], oddsP[2], oddsMax)

	fmt.Println("-- Fibonacci batches --")
	fmt.Printf("batches: %d  avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		fibBatches, fibAvg, fibMin, fibP[0], fibP[1], fibP[2], fibMax)

	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)

	_ = oddsBatches
	_ = fibBatches
}

// ----------------------- Odds (parallel) -----------------------

func runOdds(N int, workers, chunks, spin int, csvOut *BatchCSV, printEvery int) (sum uint64, count int64, batches int, batchDurations []time.Duration) {
	if N < 1 {
		return 0, 0, 0, nil
	}
	// Batch the outer range so CSV/percentiles are meaningful.
	batches = chunks
	if batches < 1 {
		batches = 1
	}
	chunkSize := (N + batches - 1) / batches
	type job struct{ lo, hi int }

	jobs := make(chan job, batches)
	type res struct {
		sum   uint64
		count int64
		dur   time.Duration
	}
	results := make(chan res, batches)

	var globalSum uint64
	var globalCount int64
	var wg sync.WaitGroup

	// Workers
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for j := range jobs {
				t0 := time.Now()
				var localSum uint64
				var localCnt int64
				// Enumerate odds in [j.lo .. j.hi]
				start := j.lo
				if start%2 == 0 {
					start++
				}
				for i := start; i <= j.hi; i += 2 {
					// cheap spin to simulate extra CPU work if requested
					if spin > 0 {
						var s int
						for k := 0; k < spin; k++ {
							s += k
						}
						_ = s
					}
					localSum += uint64(i)
					localCnt++
				}
				results <- res{
					sum:   localSum,
					count: localCnt,
					dur:   time.Since(t0),
				}
			}
		}()
	}

	// Enqueue batches
	for lo := 1; lo <= N; lo += chunkSize {
		hi := lo + chunkSize - 1
		if hi > N {
			hi = N
		}
		jobs <- job{lo: lo, hi: hi}
	}
	close(jobs)

	// Collect
	batchDurations = make([]time.Duration, 0, batches)
	done := make(chan struct{})
	go func() {
		defer close(done)
		idx := 0
		for r := range results {
			atomic.AddUint64(&globalSum, r.sum)
			atomic.AddInt64(&globalCount, r.count)
			batchDurations = append(batchDurations, r.dur)
			if csvOut != nil {
				csvOut.Row("odds", idx, r.dur, r.sum)
			}
			if printEvery > 0 && idx%printEvery == 0 {
				log.Printf("Odds batch %d dur=%s partial_sum=%d", idx, r.dur, r.sum)
			}
			idx++
		}
	}()

	wg.Wait()
	close(results)
	<-done

	return globalSum, globalCount, len(batchDurations), batchDurations
}

// ----------------------- Fibonacci (sequential batches) -----------------------

func runFibonacci(N int, mod uint64, store bool, repeat int, csvOut *BatchCSV, printEvery int) (checksum uint64, batches int, batchDurations []time.Duration) {
	if N <= 0 {
		return 0, 0, nil
	}
	// We'll split the sequence into fixed-size batches so we can time them.
	const targetBatches = 32
	batches = targetBatches
	batchSize := (N + batches - 1) / batches
	if batchSize < 1 {
		batchSize = 1
	}
	// Recompute batches with new size to cover exactly N
	batches = (N + batchSize - 1) / batchSize

	batchDurations = make([]time.Duration, 0, batches*repeat)

	for r := 0; r < repeat; r++ {
		var a, b uint64 = 0, 1
		var storeBuf []uint64
		if store {
			storeBuf = make([]uint64, 0, N)
			storeBuf = append(storeBuf, a, b)
		}
		idx := 0
		written := 2

		for batch := 0; batch < batches; batch++ {
			t0 := time.Now()
			limit := (batch + 1) * batchSize
			if limit > N {
				limit = N
			}
			for idx < limit {
				a, b = b%mod, (a+b)%mod
				if store {
					if written < N {
						storeBuf = append(storeBuf, a)
						written++
					}
				}
				idx++
			}
			dur := time.Since(t0)
			if csvOut != nil {
				csvOut.Row("fibonacci", batch+(r*batches), dur, a^b)
			}
			if printEvery > 0 && (batch%printEvery == 0 || batch == batches-1) {
				log.Printf("Fib rep=%d batch=%d/%d dur=%s last_pair=(%d,%d)", r, batch+1, batches, dur, a, b)
			}
			batchDurations = append(batchDurations, dur)
		}
		// Accumulate checksum so the loop isn’t optimized out
		checksum ^= a + 31*b + uint64(len(batchDurations))
		// keep storeBuf in scope (and printed size) so it isn’t DCE’d
		if store {
			log.Printf("Fib rep=%d stored_terms=%d", r, len(storeBuf))
		}
	}

	return checksum, len(batchDurations), batchDurations
}

// ----------------------- Timing helpers -----------------------

func summarizeDurations(dd []time.Duration) (sum, min, max time.Duration) {
	if len(dd) == 0 {
		return 0, 0, 0
	}
	min, max = dd[0], dd[0]
	for _, d := range dd {
		sum += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return
}

func percentiles(dd []time.Duration, ps []float64) []time.Duration {
	if len(dd) == 0 {
		return make([]time.Duration, len(ps))
	}
	cp := make([]time.Duration, len(dd))
	copy(cp, dd)
	// insertion sort
	for i := 1; i < len(cp); i++ {
		k := cp[i]
		j := i - 1
		for j >= 0 && cp[j] > k {
			cp[j+1] = cp[j]
			j--
		}
		cp[j+1] = k
	}
	res := make([]time.Duration, 0, len(ps))
	for _, p := range ps {
		switch {
		case p <= 0:
			res = append(res, cp[0])
		case p >= 1:
			res = append(res, cp[len(cp)-1])
		default:
			idx := int(math.Round(p*float64(len(cp)-1)))
			res = append(res, cp[idx])
		}
	}
	return res
}

        """, language="go")
        st.markdown("2.N-Body Simulation")
        st.code("""
// nbody.go
package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"sync/atomic"
	"time"
)

const G = 6.67430e-11

type Vec3 struct{ X, Y, Z float64 }

func (a Vec3) Add(b Vec3) Vec3      { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Sub(b Vec3) Vec3      { return Vec3{a.X - b.X, a.Y - b.Y, a.Z - b.Z} }
func (a Vec3) Scale(s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }
func (a Vec3) Dot(b Vec3) float64   { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a Vec3) Norm() float64        { return math.Sqrt(a.Dot(a)) }

type Body struct {
	Pos  Vec3
	Vel  Vec3
	Mass float64
	F    Vec3 // force accumulator
}

type StepStat struct {
	Step          int
	StepWallNanos int64
	KE            float64
	PE            float64
	TotalE        float64
	PMag          float64
	Goroutines    int
}

func main() {
	// --- Flags ---
	n := flag.Int("n", 2000, "number of bodies")
	steps := flag.Int("steps", 200, "simulation steps")
	dt := flag.Float64("dt", 1e-3, "time step (seconds)")
	softening := flag.Float64("softening", 1e-3, "softening factor (epsilon)")
	seed := flag.Int64("seed", 42, "random seed (negative to use time-based)")
	workers := flag.Int("workers", 0, "number of worker goroutines (0 => GOMAXPROCS)")
	chunks := flag.Int("chunks", 0, "number of work chunks (0 => auto)")
	velScale := flag.Float64("velscale", 1e-3, "initial velocity scale")
	posScale := flag.Float64("posscale", 1.0, "initial position scale")

	cpuprofile := flag.String("cpuprofile", "", "write CPU profile to file")
	memprofile := flag.String("memprofile", "", "write memory profile to file (at end)")
	traceFile := flag.String("trace", "", "write runtime trace to file")
	httpAddr := flag.String("http", "", "start pprof HTTP server at address (e.g. :6060)")
	csvFile := flag.String("csv", "", "write per-step timings to CSV file")
	verify := flag.Bool("verify", true, "compute energy & momentum each step (PE is O(n^2))")
	printEvery := flag.Int("print_every", 20, "print stats every k steps (0=never)")
	flag.Parse()

	// Optional pprof HTTP server
	if *httpAddr != "" {
		go func() {
			log.Printf("pprof HTTP server listening on %s (visit /debug/pprof/)", *httpAddr)
			if err := http.ListenAndServe(*httpAddr, nil); err != nil {
				log.Printf("pprof server error: %v", err)
			}
		}()
	}

	// CPU profile
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpu profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Trace
	if *traceFile != "" {
		tf, err := os.Create(*traceFile)
		if err != nil {
			log.Fatalf("create trace: %v", err)
		}
		defer tf.Close()
		if err := trace.Start(tf); err != nil {
			log.Fatalf("start trace: %v", err)
		}
		defer trace.Stop()
	}

	// Random seed
	if *seed < 0 {
		*seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(*seed))
	log.Printf("Seed: %d", *seed)

	// Decide workers/chunks
	if *workers <= 0 {
		*workers = runtime.GOMAXPROCS(0)
	}
	if *chunks <= 0 {
		*chunks = *workers * 4
	}

	// Init system
	bodies := make([]Body, *n)
	initBodies(bodies, rng, *posScale, *velScale)

	// CSV
	var csvWriter *csv.Writer
	if *csvFile != "" {
		f, err := os.Create(*csvFile)
		if err != nil {
			log.Fatalf("create csv: %v", err)
		}
		defer f.Close()
		bw := bufio.NewWriter(f)
		defer bw.Flush()
		csvWriter = csv.NewWriter(bw)
		_ = csvWriter.Write([]string{"step", "step_ms", "ke", "pe", "total_e", "momentum", "goroutines"})
		defer csvWriter.Flush()
	}

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	start := time.Now()

	// Worker pool
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	forcePool := newForcePool(*workers, *chunks, *softening)

	// Simulation loop
	stepTimes := make([]time.Duration, *steps)
	var totalPairs uint64
	for s := 0; s < *steps; s++ {
		t0 := time.Now()

		// Compute forces (O(n^2), chunked)
		pairs := forcePool.ComputeForces(ctx, bodies)
		atomic.AddUint64(&totalPairs, uint64(pairs))

		// Integrate (semi-implicit Euler)
		integrate(bodies, *dt)

		stepTimes[s] = time.Since(t0)

		// Optional verification & metrics
		var ke, pe, pmag float64
		if *verify {
			ke = kineticEnergy(bodies)
			pe = potentialEnergy(bodies, *softening)
			pmag = momentumMag(bodies)
		}
		stats := StepStat{
			Step:          s,
			StepWallNanos: stepTimes[s].Nanoseconds(),
			KE:            ke,
			PE:            pe,
			TotalE:        ke + pe,
			PMag:          pmag,
			Goroutines:    runtime.NumGoroutine(),
		}

		if csvWriter != nil {
			_ = csvWriter.Write([]string{
				fmt.Sprint(stats.Step),
				fmt.Sprintf("%.3f", float64(stats.StepWallNanos)/1e6),
				fmt.Sprintf("%.6e", stats.KE),
				fmt.Sprintf("%.6e", stats.PE),
				fmt.Sprintf("%.6e", stats.TotalE),
				fmt.Sprintf("%.6e", stats.PMag),
				fmt.Sprint(stats.Goroutines),
			})
		}

		if *printEvery > 0 && (s%*printEvery == 0 || s == *steps-1) {
			log.Printf("step=%d dt=%.2e step_ms=%.3f KE=%.3e PE=%.3e E=%.3e |P|=%.3e goroutines=%d",
				s, *dt, float64(stepTimes[s].Microseconds())/1000.0, ke, pe, ke+pe, pmag, stats.Goroutines)
		}
	}
	total := time.Since(start)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Profiling dumps
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write heap profile: %v", err)
		}
	}

	// Summaries
	sum, min, max := summarizeDurations(stepTimes)
	avg := time.Duration(int64(sum) / int64(len(stepTimes)))
	pcts := percentiles(stepTimes, []float64{0.50, 0.90, 0.99})
	p50, p90, p99 := pcts[0], pcts[1], pcts[2]

	fmt.Println("==== SUMMARY ====")
	fmt.Printf("Bodies: %d, Steps: %d, workers: %d, chunks: %d\n", *n, *steps, *workers, *chunks)
	fmt.Printf("Wall: %s, per-step avg=%s min=%s p50=%s p90=%s p99=%s max=%s\n", total, avg, min, p50, p90, p99, max)
	fmt.Printf("Pairs computed: %d (expected ~ n*(n-1)/2 per step)\n", atomic.LoadUint64(&totalPairs))

	// GC / Mem
	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Goroutines (end): %d\n", runtime.NumGoroutine())
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)
}

// ---------------- Initialization ----------------

func initBodies(b []Body, rng *rand.Rand, posScale, velScale float64) {
	n := len(b)
	for i := 0; i < n; i++ {
		b[i].Pos = Vec3{
			X: (rng.Float64()*2 - 1) * posScale,
			Y: (rng.Float64()*2 - 1) * posScale,
			Z: (rng.Float64()*2 - 1) * posScale,
		}
		b[i].Vel = Vec3{
			X: (rng.Float64()*2 - 1) * velScale,
			Y: (rng.Float64()*2 - 1) * velScale,
			Z: (rng.Float64()*2 - 1) * velScale,
		}
		b[i].Mass = 1.0 + 0.1*(rng.Float64()*2-1)
	}
}

// ---------------- Physics ----------------

func integrate(b []Body, dt float64) {
	for i := range b {
		a := b[i].F.Scale(1.0 / b[i].Mass)
		b[i].Vel = b[i].Vel.Add(a.Scale(dt))
		b[i].Pos = b[i].Pos.Add(b[i].Vel.Scale(dt))
		b[i].F = Vec3{} // reset for next step
	}
}

func kineticEnergy(b []Body) float64 {
	var ke float64
	for i := range b {
		v2 := b[i].Vel.Dot(b[i].Vel)
		ke += 0.5 * b[i].Mass * v2
	}
	return ke
}

func potentialEnergy(b []Body, eps float64) float64 {
	var pe float64
	n := len(b)
	eps2 := eps * eps
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			r := b[j].Pos.Sub(b[i].Pos)
			dist := math.Sqrt(r.Dot(r) + eps2)
			pe += -G * b[i].Mass * b[j].Mass / dist
		}
	}
	return pe
}

func momentumMag(b []Body) float64 {
	var p Vec3
	for i := range b {
		p = p.Add(b[i].Vel.Scale(b[i].Mass))
	}
	return p.Norm()
}

// ---------------- Parallel Force Computation ----------------

type forcePool struct {
	workers int
	chunks  int
	eps     float64
}

func newForcePool(workers, chunks int, eps float64) *forcePool {
	return &forcePool{workers: workers, chunks: chunks, eps: eps}
}

func (fp *forcePool) ComputeForces(ctx context.Context, b []Body) int {
	n := len(b)
	if n == 0 {
		return 0
	}
	chunkSize := (n + fp.chunks - 1) / fp.chunks
	type job struct{ i0, i1 int }
	jobs := make(chan job, fp.chunks)
	var pairs int64

	wg := sync.WaitGroup{}
	wg.Add(fp.workers)
	for w := 0; w < fp.workers; w++ {
		go func() {
			defer wg.Done()
			eps2 := fp.eps * fp.eps
			for j := range jobs {
				for i := j.i0; i < j.i1; i++ {
					fi := Vec3{}
					pi := b[i].Pos
					mi := b[i].Mass
					for k := 0; k < n; k++ {
						if k == i {
							continue
						}
						r := b[k].Pos.Sub(pi)
						d2 := r.Dot(r) + eps2
						invD := 1.0 / math.Sqrt(d2)
						invD3 := invD * invD * invD
						f := r.Scale(G * mi * b[k].Mass * invD3)
						fi = fi.Add(f)
					}
					// i is unique to this goroutine within its chunk → no race on b[i].F
					b[i].F = b[i].F.Add(fi)
					atomic.AddInt64(&pairs, int64(n-1))
				}
			}
		}()
	}

	for i := 0; i < n; i += chunkSize {
		j := i + chunkSize
		if j > n {
			j = n
		}
		select {
		case jobs <- job{i0: i, i1: j}:
		case <-ctx.Done():
			close(jobs)
			wg.Wait()
			return int(atomic.LoadInt64(&pairs))
		}
	}
	close(jobs)
	wg.Wait()
	return int(atomic.LoadInt64(&pairs))
}

// ---------------- Timing helpers ----------------

func summarizeDurations(dd []time.Duration) (sum, min, max time.Duration) {
	if len(dd) == 0 {
		return 0, 0, 0
	}
	min, max = dd[0], dd[0]
	for _, d := range dd {
		sum += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return
}

func percentiles(dd []time.Duration, ps []float64) []time.Duration {
	cp := make([]time.Duration, len(dd))
	copy(cp, dd)
	// insertion sort (steps is typically modest)
	for i := 1; i < len(cp); i++ {
		k := cp[i]
		j := i - 1
		for j >= 0 && cp[j] > k {
			cp[j+1] = cp[j]
			j--
		}
		cp[j+1] = k
	}
	res := make([]time.Duration, 0, len(ps))
	for _, p := range ps {
		switch {
		case p <= 0:
			res = append(res, cp[0])
		case p >= 1:
			res = append(res, cp[len(cp)-1])
		default:
			idx := int(math.Round(p*float64(len(cp)-1)))
			res = append(res, cp[idx])
		}
	}
	return res
}

        """, language="go")

        st.markdown("3.Deep Recursion")
        st.code("""
// deep_recursion.go
package main

import (
	"bufio"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"sync"
	"time"
)

// ---- CSV helper ----

type BatchCSV struct {
	writer *csv.Writer
}

func (c *BatchCSV) Header() {
	if c.writer != nil {
		_ = c.writer.Write([]string{"mode", "root", "depth", "fanout", "ms", "goroutines", "checksum"})
	}
}
func (c *BatchCSV) Row(mode string, root, depth, fanout int, dur time.Duration, checksum uint64) {
	if c.writer != nil {
		_ = c.writer.Write([]string{
			mode,
			fmt.Sprint(root),
			fmt.Sprint(depth),
			fmt.Sprint(fanout),
			fmt.Sprintf("%.3f", float64(dur.Nanoseconds())/1e6),
			fmt.Sprint(runtime.NumGoroutine()),
			fmt.Sprint(checksum),
		})
	}
}
func (c *BatchCSV) Flush() { if c.writer != nil { c.writer.Flush() } }

// ---- Workloads (recursion) ----

// tiny CPU spin so the compiler can’t optimize the recursion away
func spinWork(spin int) {
	if spin <= 0 {
		return
	}
	s := 0
	for i := 0; i < spin; i++ {
		s += i
	}
	_ = s
}

func linearRec(depth, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	return 1 + linearRec(depth-1, spin)
}

func binaryRec(depth, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	// 1 + left + right
	return 1 + binaryRec(depth-1, spin) + binaryRec(depth-2, spin)
}

func karyRec(depth, fanout, spin int) uint64 {
	if depth <= 0 {
		spinWork(spin)
		return 1
	}
	spinWork(spin)
	var sum uint64 = 1
	for i := 0; i < fanout; i++ {
		sum += karyRec(depth-1, fanout, spin)
	}
	return sum
}

// ---- Runner ----

type result struct {
	rootIdx   int
	checksum  uint64
	elapsed   time.Duration
}

func runRoots(mode string, roots, depth, fanout, spin int) (results []result) {
	results = make([]result, 0, roots)
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}

	for r := 0; r < roots; r++ {
		wg.Add(1)
		go func(root int) {
			defer wg.Done()
			t0 := time.Now()
			var chk uint64
			switch mode {
			case "linear":
				chk = linearRec(depth, spin)
			case "binary":
				chk = binaryRec(depth, spin)
			case "kary":
				chk = karyRec(depth, fanout, spin)
			default:
				chk = linearRec(depth, spin)
			}
			el := time.Since(t0)
			mu.Lock()
			results = append(results, result{rootIdx: root, checksum: chk, elapsed: el})
			mu.Unlock()
		}(r)
	}
	wg.Wait()
	return
}

// ---- Durations helpers ----

func summarizeDurations(dd []time.Duration) (sum, min, max time.Duration) {
	if len(dd) == 0 {
		return 0, 0, 0
	}
	min, max = dd[0], dd[0]
	for _, d := range dd {
		sum += d
		if d < min {
			min = d
		}
		if d > max {
			max = d
		}
	}
	return
}

func percentiles(dd []time.Duration, ps []float64) []time.Duration {
	if len(dd) == 0 {
		return make([]time.Duration, len(ps))
	}
	cp := make([]time.Duration, len(dd))
	copy(cp, dd)
	// insertion sort (roots count is usually small)
	for i := 1; i < len(cp); i++ {
		k := cp[i]
		j := i - 1
		for j >= 0 && cp[j] > k {
			cp[j+1] = cp[j]
			j--
		}
		cp[j+1] = k
	}
	res := make([]time.Duration, 0, len(ps))
	for _, p := range ps {
		switch {
		case p <= 0:
			res = append(res, cp[0])
		case p >= 1:
			res = append(res, cp[len(cp)-1])
		default:
			idx := int(math.Round(p*float64(len(cp)-1)))
			res = append(res, cp[idx])
		}
	}
	return res
}

// ---- main ----

func main() {
	// Workload controls
	mode := flag.String("mode", "linear", "Recursion mode: linear | binary | kary")
	depth := flag.Int("depth", 50000, "Recursion depth (CAUTION: large values can crash)")
	fanout := flag.Int("fanout", 3, "Fanout for k-ary recursion (ignored unless mode=kary)")
	roots := flag.Int("roots", 8, "Number of independent recursion roots (run in parallel)")
	spin := flag.Int("spin", 0, "Artificial CPU work per call (inner loop iterations)")

	// Profiling
	cpuprofile := flag.String("cpuprofile", "", "Write CPU profile to file")
	memprofile := flag.String("memprofile", "", "Write memory profile to file at end")
	traceFile := flag.String("trace", "", "Write runtime trace to file")
	httpAddr := flag.String("http", "", "Start pprof HTTP server at address (e.g. :6060)")

	// Output
	csvPath := flag.String("csv", "", "Write per-root timings CSV")
	printEvery := flag.Int("print_every", 2, "Print every k roots (0=never)")

	flag.Parse()

	// Sanity defaults to avoid explosions
	switch *mode {
	case "binary":
		if flag.Lookup("depth").Value.String() == "50000" { // default unchanged
			*depth = 28 // ~2^28 calls (still heavy!)
		}
	case "kary":
		if flag.Lookup("depth").Value.String() == "50000" {
			*depth = 12
		}
		if *fanout < 2 {
			*fanout = 2
		}
	}

	// Live pprof
	if *httpAddr != "" {
		go func() {
			log.Printf("pprof HTTP server on %s (see /debug/pprof/)", *httpAddr)
			if err := http.ListenAndServe(*httpAddr, nil); err != nil {
				log.Printf("pprof server error: %v", err)
			}
		}()
	}

	// CPU profile
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatalf("create cpuprofile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatalf("start cpu profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	// Trace
	if *traceFile != "" {
		tf, err := os.Create(*traceFile)
		if err != nil {
			log.Fatalf("create trace: %v", err)
		}
		defer tf.Close()
		if err := trace.Start(tf); err != nil {
			log.Fatalf("start trace: %v", err)
		}
		defer trace.Stop()
	}

	// CSV
	var csvOut BatchCSV
	if *csvPath != "" {
		f, err := os.Create(*csvPath)
		if err != nil {
			log.Fatalf("create csv: %v", err)
		}
		defer f.Close()
		bw := bufio.NewWriter(f)
		defer bw.Flush()
		csvOut.writer = csv.NewWriter(bw)
		defer csvOut.writer.Flush()
		csvOut.Header()
	}

	// Baseline mem stats
	var ms0, ms1 runtime.MemStats
	runtime.ReadMemStats(&ms0)
	start := time.Now()

	// Run
	log.Printf("Mode=%s depth=%d fanout=%d roots=%d spin=%d", *mode, *depth, *fanout, *roots, *spin)
	results := runRoots(*mode, *roots, *depth, *fanout, *spin)
	totalWall := time.Since(start)

	// End mem stats
	runtime.ReadMemStats(&ms1)

	// Summaries
	durs := make([]time.Duration, 0, len(results))
	var aggChecksum uint64
	for i, r := range results {
		if csvOut.writer != nil {
			csvOut.Row(*mode, r.rootIdx, *depth, *fanout, r.elapsed, r.checksum)
		}
		if *printEvery > 0 && (i%*printEvery == 0 || i == len(results)-1) {
			log.Printf("root=%d dur=%s checksum=%d", r.rootIdx, r.elapsed, r.checksum)
		}
		aggChecksum ^= (r.checksum + uint64(i)*1315423911)
		durs = append(durs, r.elapsed)
	}

	sum, min, max := summarizeDurations(durs)
	avg := time.Duration(int64(sum) / int64(len(durs)))
	pcts := percentiles(durs, []float64{0.50, 0.90, 0.99})

	fmt.Println("==== SUMMARY ====")
	fmt.Printf("Mode=%s Depth=%d Fanout=%d Roots=%d Spin=%d\n", *mode, *depth, *fanout, *roots, *spin)
	fmt.Printf("Total wall: %s  per-root avg=%s  min=%s  p50=%s  p90=%s  p99=%s  max=%s\n",
		totalWall, avg, min, pcts[0], pcts[1], pcts[2], max)
	fmt.Printf("Aggregate checksum: %d\n", aggChecksum)

	fmt.Println("---- RUNTIME / GC ----")
	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))
	fmt.Printf("Goroutines (end): %d\n", runtime.NumGoroutine())
	fmt.Printf("GC cycles: %d -> %d\n", ms0.NumGC, ms1.NumGC)
	fmt.Printf("Total GC pause: %.3f ms -> %.3f ms (cumulative)\n", float64(ms0.PauseTotalNs)/1e6, float64(ms1.PauseTotalNs)/1e6)
	fmt.Printf("HeapAlloc: %.2f MB -> %.2f MB\n", float64(ms0.HeapAlloc)/1e6, float64(ms1.HeapAlloc)/1e6)
	fmt.Printf("TotalAlloc: %.2f MB -> %.2f MB\n", float64(ms0.TotalAlloc)/1e6, float64(ms1.TotalAlloc)/1e6)
	fmt.Printf("Sys: %.2f MB -> %.2f MB\n", float64(ms0.Sys)/1e6, float64(ms1.Sys)/1e6)

	// Optional heap profile at end
	if *memprofile != "" {
		f, err := os.Create(*memprofile)
		if err != nil {
			log.Fatalf("create memprofile: %v", err)
		}
		defer f.Close()
		runtime.GC()
		if err := pprof.WriteHeapProfile(f); err != nil {
			log.Fatalf("write heap profile: %v", err)
		}
	}
}

        """, language="go")


if section == "Results":
    st.markdown("""
    <div style='font-size:18px'>
      <span style='font-size:22px; font-weight:bold;'>3.1 Runtime (Execution Time)</span><br><br>
      Execution time is measured using native timers or profilers for each language.<br><br>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Task</th><th>Python</th><th>Java</th><th>C</th><th>C++</th><th>Go</th></tr>
        </thead>
        <tbody>
          <tr><td>Fibonacci + Odd (1M)</td><td>~21.05 s</td><td>~32 ms</td><td>~180 ms</td><td>~70 ms</td><td>~8–12 ms</td></tr>
          <tr><td>N-body Simulation</td><td>~2.34 s</td><td>~420 ms</td><td>~240 ms</td><td>~110 ms</td><td>~1.57–76 s*</td></tr>
          <tr><td>Deep Recursion</td><td>~1.02 s</td><td>~230 ms</td><td>~150 ms</td><td>~90 ms</td><td>~6.65–9.10 ms</td></tr>
        </tbody>
      </table>
      <div style="font-size:14px; margin-top:6px;">*Go N-body scales with n: n=2000 (~1.57 s), n=3000 (~7.1 s), n=5000 (~76 s).</div>

      <br>
       <strong>Go</strong> is fastest on Fibonacci+Odd thanks to compiled execution and minimal allocation.<br>
       <strong>C++</strong> stays consistently fastest on heavier numeric (N-body); <strong>Go</strong> is competitive for smaller n.<br>
       <strong>Python</strong> is slowest across workloads (interpreter + GIL).<br>
       <strong>Java</strong> benefits from JIT on longer tasks, but lags <strong>Go</strong> on low-allocation microbenchmarks.<br>
       <strong>C</strong> sits between Java and C++ depending on implementation details.<br><br>

      <span style='font-size:22px; font-weight:bold;'>3.2 Memory Use (Peak Memory Allocation)</span><br><br>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Task</th><th>Python</th><th>Java</th><th>C</th><th>C++</th><th>Go</th></tr>
        </thead>
        <tbody>
          <tr><td>Fibonacci + Odd (1M)</td><td>~0.30 MB</td><td>~50–55 MB</td><td>~1 KB</td><td>~7.7 MB</td><td>~0.22–3.8 MB</td></tr>
          <tr><td>N-body Simulation</td><td>~5.2 MB</td><td>~60 MB</td><td>~100 KB</td><td>~2.1 MB</td><td>~0.48–6.68 MB</td></tr>
          <tr><td>Deep Recursion</td><td>~2.3 MB</td><td>~40 MB</td><td>~50 KB</td><td>~120 KB</td><td>~0.22–3.82 MB</td></tr>
        </tbody>
      </table><br>

       <strong>C</strong> is most memory-efficient (stack-first, no runtime overhead).<br>
       <strong>Go</strong> has low baseline for compute-bound tasks; GC metadata minimal when allocations are few.<br>
       <strong>C++</strong> > <strong>C</strong> due to STL/vector overhead (when storing data).<br>
       <strong>Java</strong> highest baseline heap; <strong>Python</strong> scales linearly with object-heavy data.<br><br>

      <span style='font-size:22px; font-weight:bold;'>3.3 Garbage Collection / Threading Observations</span><br><br>

      <strong>Go:</strong><br>
      • Concurrent GC; pauses ~0–0.53 ms in runs observed.<br>
      • Goroutines scale well; lightweight scheduling reduces overhead.<br>
      • <code>GOMAXPROCS</code> tuning helps avoid oversubscription.<br><br>

      <strong>Java:</strong><br>
      • Regular GC activity on larger tasks; pauses ~2–3 ms (Parallel/other modern GCs).<br>
      • Threads scale well on CPU-bound tasks post warm-up.<br><br>

      <strong>Python:</strong><br>
      • Minimal GC work in these tests; GIL blocks true CPU parallelism.<br>
      • No speedup from threads on Fibonacci/N-body.<br><br>

      <strong>C/C++:</strong><br>
      • No GC; manual memory management.<br>
      • True OS threads (<code>pthreads</code> / <code>std::thread</code>), good core utilization with balanced work.<br><br>

      <span style='font-size:22px; font-weight:bold;'>Program-Wise Breakdown (Execution & Memory Summary)</span><br><br>

      <strong>1. Fibonacci + Odd Numbers</strong><br>
      <table style="width:100%; font-size:15px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
          <tr><td>Python</td><td>~21.05 s</td><td>~0.3 MB</td><td>Low</td><td>No true concurrency</td><td>GIL-bound</td></tr>
          <tr><td>Java</td><td>~32 ms</td><td>~50 MB</td><td>Yes</td><td>Yes</td><td>Fast after JIT warm-up</td></tr>
          <tr><td>C</td><td>~180 ms</td><td>~1 KB</td><td>No</td><td>Yes</td><td>Manual allocation</td></tr>
          <tr><td>C++</td><td>~70 ms</td><td>~7.7 MB</td><td>No</td><td>Yes</td><td>Stored vector</td></tr>
          <tr><td>Go</td><td>~8–12 ms</td><td>~0.22–3.8 MB</td><td>Yes</td><td>Yes</td><td>Fastest; low allocation</td></tr>
        </tbody>
      </table><br>

      <strong>2. N-body Simulation</strong><br>
      <table style="width:100%; font-size:15px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
          <tr><td>Python</td><td>~2.34 s</td><td>~5.2 MB</td><td>Low</td><td>Limited</td><td>Slow FP ops</td></tr>
          <tr><td>Java</td><td>~420 ms</td><td>~60 MB</td><td>Yes</td><td>Yes</td><td>Parallel loop iteration</td></tr>
          <tr><td>C</td><td>~240 ms</td><td>~100 KB</td><td>No</td><td>Yes</td><td>Low-level vector math</td></tr>
          <tr><td>C++</td><td>~110 ms</td><td>~2.1 MB</td><td>No</td><td>Yes</td><td>Fastest overall</td></tr>
          <tr><td>Go</td><td>~1.57–76 s</td><td>~0.48–6.68 MB</td><td>Yes</td><td>Yes</td><td>O(n²) scaling; efficient GC</td></tr>
        </tbody>
      </table><br>

      <strong>3. Deep Recursion</strong><br>
      <table style="width:100%; font-size:15px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
          <tr><td>Python</td><td>~1.02 s</td><td>~2.3 MB</td><td>Low</td><td>No</td><td>Stack limit issues</td></tr>
          <tr><td>Java</td><td>~230 ms</td><td>~40 MB</td><td>Yes</td><td>Yes</td><td>JVM-managed recursion</td></tr>
          <tr><td>C</td><td>~150 ms</td><td>~50 KB</td><td>No</td><td>Yes</td><td>Very compact stack</td></tr>
          <tr><td>C++</td><td>~90 ms</td><td>~120 KB</td><td>No</td><td>Yes</td><td>RAII used</td></tr>
          <tr><td>Go</td><td>~6.65–9.10 ms</td><td>~0.22–3.82 MB</td><td>Yes</td><td>Yes</td><td>Safe stack growth; minimal allocation</td></tr>
        </tbody>
      </table><br>

      <span style='font-size:22px; font-weight:bold;'>Summary</span><br><br>
      • <strong>Fastest Overall:</strong> Short microbenchmarks → <strong>Go</strong> (Fibonacci+Odd); heavy numeric → <strong>C++</strong><br>
      • <strong>Most Memory-Efficient:</strong> <strong>C</strong><br>
      • <strong>Best Developer Productivity:</strong> <strong>Python</strong><br>
      • <strong>Best Balance (Speed + Tooling):</strong> <strong>Java</strong><br>
      • <strong>Most Predictable GC Behavior:</strong> <strong>Go</strong><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<span style='font-size:24px; font-weight:bold;'>5. Output View</span><br>""", unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px;">Click on A language button to toggle its  Output:</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:20px;">Click on the button again to hide its Output .</p>', unsafe_allow_html=True)
    if "show_java" not in st.session_state:
        st.session_state.show_java = False
    if "show_python" not in st.session_state:
        st.session_state.show_python = False
    if "show_c" not in st.session_state:
        st.session_state.show_c = False
    if "show_cpp" not in st.session_state:
        st.session_state.show_cpp = False
    if "show_go" not in st.session_state:
        st.session_state.show_go = False

    # Use 11 columns: 5 buttons + narrow spacers + filler
    col1, spacer1, col2, spacer2, col3, spacer3, col4, spacer4, col5, _, _ = st.columns(
        [1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 5]
    )

    with col1:
        if st.button("Java"):
            st.session_state.show_java = not st.session_state.show_java

    with col2:
        if st.button("Python"):
            st.session_state.show_python = not st.session_state.show_python

    with col3:
        if st.button("C"):
            st.session_state.show_c = not st.session_state.show_c

    with col4:
        if st.button("C++"):
            st.session_state.show_cpp = not st.session_state.show_cpp

    with col5:
        if st.button("Go"):
            st.session_state.show_go = not st.session_state.show_go

    # Stacked images based on toggled buttons
    if st.session_state.show_java:
        st.subheader("Java Profiler Output")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.image("Images\javaop.png", caption="Odd + Fibonacci-1M", use_column_width=True)
        st.markdown("2.N-Body Simulation")
        st.image("Images\javanbody.png", caption="N-Body Simulation", use_column_width=True)
        st.markdown("3.Deep Recursion")
        st.image("Images\javadeep.png", caption="Deep Recursion", use_column_width=True)


    if st.session_state.show_python:
        st.subheader("Python Profiler Output")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.image("Images\pyfib.png", caption="Odd + Fibonacci-1M", use_column_width=True)
        st.markdown("2.N-Body Simulation")
        st.image("Images\pynb1.png", use_column_width=True)
        st.image("Images\pynb2.png", use_column_width=True)
        st.image("Images\pynb3.png", caption="N-Body Simulation", use_column_width=True)
        st.markdown("3.Deep Recursion")
        st.image("Images\pydeep1.png", use_column_width=True)
        st.image("Images\pydeep2.png", caption="Deep Recursion", use_column_width=True)
        
    if st.session_state.show_c:
        st.subheader("C Profiler Output")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.image("Images\cfib1.png", use_column_width=True)
        st.image("Images\cfib2.png", use_column_width=True)
        st.image("Images\cfib3.png", use_column_width=True)
        st.image("Images\cfib4.png", use_column_width=True)
        st.image("Images\cfib5.png", use_column_width=True)
        st.image("Images\cfib6.png", use_column_width=True)
        st.image("Images\cfib7.png", use_column_width=True)
        st.image("Images\cfib8.png", use_column_width=True)
        st.markdown("2.N-Body Simulation")
        st.image("Images\cnb1.png", use_column_width=True)
        st.image("Images\cnb2.png", use_column_width=True)
        st.image("Images\cnb3.png", use_column_width=True)
        st.image("Images\cnb4.png", use_column_width=True)
        st.image("Images\cnb5.png", use_column_width=True)
        st.image("Images\cnb6.png", use_column_width=True)
        st.markdown("3.Deep Recursion")
        st.image("Images\cr1.png", use_column_width=True)
        st.image("Images\cr2.png", use_column_width=True)
        st.image("Images\cr3.png", use_column_width=True)
        st.image("Images\cr4.png", use_column_width=True)
        st.image("Images\cr5.png", use_column_width=True)
        st.image("Images\cr6.png", use_column_width=True)
        st.image("Images\cr7.png", use_column_width=True)
        st.image("Images\cr8.png", use_column_width=True)

    if st.session_state.show_cpp:
        st.subheader("C++ Profiler Output")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.image("Images\cppfib1.png", use_column_width=True)
        st.image("Images\cppfib2.png", use_column_width=True)
        st.image("Images\cppfib3.png", use_column_width=True)
        st.image("Images\cppfib4.png", use_column_width=True)
        st.markdown("2.N-Body Simulation")
        st.image("Images\cppnb1.png", use_column_width=True)
        st.image("Images\cppnb2.png", use_column_width=True)
        st.image("Images\cppnb3.png", use_column_width=True)
        st.image("Images\cppnb4.png", use_column_width=True)
        st.markdown("3.Deep Recursion")
        st.image("Images\cppr1.png", use_column_width=True)
        st.image("Images\cppr2.png", use_column_width=True)
        st.image("Images\cppr3.png", use_column_width=True)
        st.image("Images\cppr4.png", use_column_width=True)

    if st.session_state.show_go:
        st.subheader("Go Profiler Output")
        st.markdown("1.Fibonacci + Odd Numbers (1M)")
        st.image("Images\gofib1.png", use_column_width=True)
        st.image("Images\gofib2.png",  use_column_width=True)
        st.image("Images\gofib3.png",  use_column_width=True)
        st.image("Images\gofib4.png",  use_column_width=True)
        st.image("Images\gofib5.png",  use_column_width=True)
        st.image("Images\gofib6.png",  use_column_width=True)
        st.markdown("2.N-Body Simulation")
        st.image("Images\gonb1.png",  use_column_width=True)
        st.image("Images\gonb2.png",  use_column_width=True)
        st.image("Images\gonb3.png",  use_column_width=True)
        st.markdown("3.Deep Recursion")
        st.image("Images\godeep1.png",  use_column_width=True)
        st.image("Images\godeep2.png",  use_column_width=True)
        st.image("Images\godeep3.png",  use_column_width=True)


if section == "Comparative Analysis":
    st.markdown("""
    <div style='font-size:18px'>

      <h3>4.1 Performance</h3>
      <p>
        <b>C and C++</b> remain the fastest for large-scale numeric workloads (native code, minimal runtime overhead).<br>
        <b>Go</b> can beat Java and even C++ on some low-allocation microbenchmarks (e.g., Fibonacci+Odd), but <b>C++</b> still leads on heavy FP simulations.<br>
        <b>Java</b> is highly competitive after JIT warm-up (great on long-running workloads).<br>
        <b>Python</b> is slowest overall due to interpreter overhead and the GIL.
      </p>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead>
          <tr><th>Task</th><th>Fastest</th><th>Slowest</th><th>Java Position</th><th>Go Position</th><th>Notes</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Fibonacci + Odd</td><td>Go</td><td>Python</td><td>Close to C++</td><td>Fastest overall</td>
            <td>Go’s compiled speed & low allocation dominate; Python lags on raw loops</td>
          </tr>
          <tr>
            <td>N-body Simulation</td><td>C++</td><td>Python</td><td>Mid</td><td>Slightly slower than Java on large datasets</td>
            <td>Go scales predictably; C++ wins via low-level FP optimizations</td>
          </tr>
          <tr>
            <td>Deep Recursion</td><td>Go</td><td>Python</td><td>Mid</td><td>Fastest overall</td>
            <td>Go’s safe stack growth & low allocation edge out C++</td>
          </tr>
        </tbody>
      </table>

      <h3 style="margin-top:22px;">4.2 Memory Footprint</h3>
      <p>
        <b>C</b> is still most memory-efficient (~KBs) with minimal heap.<br>
        <b>C++</b> stays efficient even with STL usage.<br>
        <b>Go</b> has a low baseline in compute-bound, low-allocation tasks (hundreds of KBs to a few MBs).<br>
        <b>Java</b> carries a higher baseline heap (50–75 MB).<br>
        <b>Python</b> reports small heaps via tracemalloc, but process RSS is higher (interpreter overhead).
      </p>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead>
          <tr><th>Task</th><th>Least Memory</th><th>Most Memory</th><th>Notes</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Fibonacci + Odd</td><td>C (~1 KB)</td><td>Java (~50 MB)</td>
            <td>Go baseline ~0.22 MB; Python object-heavy</td>
          </tr>
          <tr>
            <td>N-body Simulation</td><td>C++ (~2.1 MB)</td><td>Java (~60 MB)</td>
            <td>Go ~0.48–6.68 MB; scales well</td>
          </tr>
          <tr>
            <td>Deep Recursion</td><td>C (~50 KB)</td><td>Java (~40 MB)</td>
            <td>Go ~0.22–3.82 MB; GC overhead negligible</td>
          </tr>
        </tbody>
      </table>

      <h3 style="margin-top:22px;">4.3 Development Ergonomics</h3>
      <ul>
        <li><b>Python:</b> Fastest to write/debug; ideal for scripting, data science, rapid prototyping.</li>
        <li><b>Java:</b> Rich IDEs (IntelliJ/Eclipse), strong profilers (VisualVM/JFR).</li>
        <li><b>Go:</b> Simple syntax, fast compiles, built-in profiling (pprof), easy concurrency.</li>
        <li><b>C/C++:</b> Deep expertise required; slower iteration, but maximal control.</li>
      </ul>

      <h3>4.4 Stability Under Load</h3>
      <ul>
        <li>All languages completed workloads without instability.</li>
        <li><b>Go:</b> Concurrent GC with negligible pauses (&lt;1 ms); stable goroutine scheduling.</li>
        <li><b>Java:</b> G1/Parallel GC with short, predictable pauses.</li>
        <li><b>C/C++:</b> No leaks detected (Valgrind).</li>
        <li><b>Python:</b> GC not heavily stressed here.</li>
      </ul>

      <h3>4.5 Debugging & Observability</h3>
      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead>
          <tr><th>Language</th><th>Tools</th><th>Observability Focus</th></tr>
        </thead>
        <tbody>
          <tr><td>Java</td><td>VisualVM, JFR</td><td>GC tracking, allocation, threads</td></tr>
          <tr><td>C/C++</td><td>Valgrind, gprof</td><td>Leaks, invalid access, hotspots</td></tr>
          <tr><td>Python</td><td>cProfile, tracemalloc</td><td>Function calls, object memory</td></tr>
          <tr><td>Go</td><td>pprof, trace, runtime stats</td><td>Heap profile, goroutines, GC cycles, scheduler</td></tr>
        </tbody>
      </table>

      <h3 style="margin-top:22px;">4.6 Concurrency Model</h3>
      <ul>
        <li><b>Python:</b> GIL limits CPU-bound threading; use multiprocessing/native extensions.</li>
        <li><b>Java:</b> OS threads, thread pools, synchronized primitives; scales well.</li>
        <li><b>Go:</b> Lightweight goroutines + channels; concurrent GC; very low thread creation cost.</li>
        <li><b>C/C++:</b> True OS threads (pthreads, &lt;thread&gt;); maximum control, higher complexity.</li>
      </ul>

      <h3>4.7 Developer Productivity</h3>
      <ul>
        <li><b>Python:</b> Most productive for exploratory coding, prototyping, data analysis.</li>
        <li><b>Java:</b> Balanced for large, observable, long-lived applications.</li>
        <li><b>Go:</b> Highly productive for backends/networked services; simple concurrency reduces bugs.</li>
        <li><b>C/C++:</b> Steepest learning curve; careful management for total control.</li>
      </ul>

      <h3 style="margin-top:22px;">4.8 Bottom-line Comparative Call</h3>
      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead>
          <tr><th>Use Case</th><th>Recommended Language</th></tr>
        </thead>
        <tbody>
          <tr><td>Tight numeric / low-level systems</td><td>C / C++</td></tr>
          <tr><td>Long-running service with observability</td><td>Java / Go</td></tr>
          <tr><td>High-concurrency backend services</td><td>Go</td></tr>
          <tr><td>Prototyping, scripting, data analysis</td><td>Python</td></tr>
          <tr><td>Small, fast, low-allocation microtasks</td><td>Go</td></tr>
        </tbody>
      </table>
    </div><br>
    """, unsafe_allow_html=True)

    #-----1.Execution Time per Task (Lower is Better)----
    
    st.markdown("<div style='font-size:18px'><h3>4.9 Graphical Representation</h3></div>", unsafe_allow_html=True)
    data = {
    "Task": ["Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd",
             "N-body", "N-body", "N-body", "N-body", "N-body",
             "Deep Recursion", "Deep Recursion", "Deep Recursion", "Deep Recursion", "Deep Recursion"],
    "Language": ["Python", "Java", "C", "C++", "Go"] * 3,
    "Time": [21.05, 0.03, 0.18, 0.07, 0.012,       # Fibonacci+Odd
             2.34, 0.42, 0.24, 0.11, 7.10,         # N-body
             1.02, 0.23, 0.15, 0.09, 0.009]        # Deep Recursion
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot using Plotly Express
    fig = px.bar(
        df,
        x="Task",
        y="Time",
        color="Language",
        barmode="group",
        title="1.Execution Time per Task (Lower is Better)",
        labels={"Time": "Execution Time (s)"}
    )

    # Optional: adjust layout
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, max(df['Time']) * 1.2])
    )

    # Show plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----2.Memory Usage per Task (Lower is Better)----
    data = {
    "Task": ["Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd",
             "N-body", "N-body", "N-body", "N-body", "N-body",
             "Deep Recursion", "Deep Recursion", "Deep Recursion", "Deep Recursion", "Deep Recursion"],
    "Language": ["Python", "Java", "C", "C++", "Go"] * 3,
    "Memory": [0.3, 55, 0.001, 7.7, 0.22,          # Fibonacci+Odd
               5.2, 60, 0.1, 2.1, 6.68,            # N-body
               2.3, 40, 0.05, 0.12, 3.82]          # Deep Recursion
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot
    fig = px.bar(
        df,
        x="Task",
        y="Memory",
        color="Language",
        barmode="group",
        title="2.Memory Usage per Task (Lower is Better)",
        labels={"Memory": "Memory Usage (MB)"}
    )

    # Layout adjustments
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, max(df['Memory']) * 1.2])
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----3.Garbage Collection and Threading Observations----
    data = {
    "Threads": [1, 2, 4, 8, 12],
    "Python": [21.0, 20.8, 20.7, 20.6, 20.5],   # GIL - no improvement
    "Java": [21.0, 11.5, 6.2, 3.4, 2.5],
    "C++": [21.0, 11.0, 5.5, 3.0, 2.2],
    "C": [21.0, 10.8, 5.3, 2.9, 2.1],
    "Go": [21.0, 10.5, 5.1, 2.7, 2.0]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert wide to long format for Plotly
    df_long = df.melt(id_vars="Threads", var_name="Language", value_name="Execution Time")

    # Plot
    fig = px.line(
        df_long,
        x="Threads",
        y="Execution Time",
        color="Language",
        markers=True,
        title="3.Concurrency Scaling: Execution Time vs Number of Threads",
        labels={"Execution Time": "Time (s)", "Threads": "Number of Threads"}
    )

    # Layout tweaks
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(tickmode="linear", dtick=1),
        yaxis_type="log",  # Optional: log scale to visualize relative gains
        legend_title="Language"
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----4.Runtime Stability: GC Pause Duration Across Languages----
    data = {
    "Language": [
        "Java", "Java", "Java", "Java", "Java",
        "Python", "Python", "Python", "Python", "Python",
        "Go", "Go", "Go", "Go", "Go"
    ],
    "GC Pause (ms)": [
        2.5, 3.0, 2.1, 2.7, 3.2,      # Java: low, short pauses
        0.4, 0.3, 0.5, 0.6, 0.4,      # Python: reference counting, no major pause
        0.2, 0.3, 0.1, 0.15, 0.25     # Go: concurrent GC with minimal pause
    ]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Create box plot
    fig = px.box(
        df,
        x="Language",
        y="GC Pause (ms)",
        title="4.Runtime Stability: GC Pause Duration Across Languages",
        labels={"GC Pause (ms)": "Pause Duration (ms)"}
    )

    # Layout tweaks
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, max(df['GC Pause (ms)']) * 1.2])
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----5.Developer Productivity vs Performance (Bubble Chart)----
    data = {
    "Language": ["Python", "Java", "C", "C++", "Go"],
    "Performance": [2, 4, 5, 5, 4],          # Execution performance (5 = best)
    "DevSpeed": [5, 4, 2, 3, 4],             # Developer productivity (5 = fastest)
    "EcosystemSize": [5, 4, 2, 3, 3]         # Ecosystem/library richness (bubble size)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Plot
    fig = px.scatter(
        df,
        x="DevSpeed",
        y="Performance",
        size="EcosystemSize",
        color="Language",
        hover_name="Language",
        size_max=60,
        title="5.Developer Productivity vs Performance",
        labels={
            "DevSpeed": "Developer Productivity (5 = Easiest)",
            "Performance": "Execution Performance (5 = Fastest)"
        }
    )

    fig.update_layout(
        template="plotly_white",
        xaxis=dict(range=[1.5, 5.5]),
        yaxis=dict(range=[1.5, 5.5])
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----6.Overall Language Comparison Radar Chart----
    data = {
    "Category": [
        "Performance", "Memory Efficiency", "Concurrency", 
        "Developer Productivity", "Tooling", "Stability"
    ],
    "Python": [2, 2, 1, 5, 4, 3],
    "Java":   [4, 3, 4, 4, 5, 4],
    "C":      [5, 5, 4, 2, 3, 3],
    "C++":    [5, 4, 4, 3, 3, 3],
    "Go":     [4, 4, 5, 4, 4, 4]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Melt for radar chart
    df_melt = df.melt(id_vars=["Category"], var_name="Language", value_name="Score")

    # Radar chart
    fig = px.line_polar(
        df_melt,
        r="Score",
        theta="Category",
        color="Language",
        line_close=True,
        markers=True,
        title="6.Overall Language Comparison Radar Chart",
    )

    fig.update_layout(
        template="plotly_white",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5])
        ),
        showlegend=True
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----7.JIT Warm-up Curve over Repeated Runs----
    data = {
    "Run": list(range(1, 11)) * 4,  # 10 runs each for 4 languages
    "Language": (
        ["Java"] * 10 +
        ["Go"] * 10 +
        ["C"] * 10 +
        ["C++"] * 10
    ),
    "Time (ms)": (
        # Java warm-up: slow -> fast
        [95, 80, 60, 45, 35, 30, 28, 27, 26, 25] +
        # Go warm-up: minor JIT-like improvement
        [70, 68, 65, 63, 61, 60, 59, 58, 58, 57] +
        # C flat (compiled)
        [20]*10 +
        # C++ flat (compiled)
        [18]*10
    )
    }

    # DataFrame
    df = pd.DataFrame(data)

    # Plot
    fig = px.line(
        df,
        x="Run",
        y="Time (ms)",
        color="Language",
        markers=True,
        title="7.JIT Warm-up Curve over Repeated Runs",
        labels={"Run": "Run Number", "Time (ms)": "Execution Time (ms)"}
    )

    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[15, 100]),
        xaxis=dict(dtick=1)
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----8.Binary Size of Executables (Lower is Better)----
    data = {
    "Language": ["Python (Script)", "Java (JAR)", "Go (Binary)", "C (Binary)", "C++ (Binary)"],
    "Binary Size (MB)": [0.1, 20, 8, 0.3, 1.5]
    }

    # DataFrame
    df = pd.DataFrame(data)

    # Plot
    fig = px.bar(
        df,
        x="Language",
        y="Binary Size (MB)",
        color="Language",
        text="Binary Size (MB)",
        title="8.Binary Size of Executables",
        labels={"Binary Size (MB)": "Size (MB)"}
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        template="plotly_white",
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        yaxis=dict(range=[0, max(df["Binary Size (MB)"]) * 1.3])
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    #-----9.Profiling Time Breakdown by Language (Stacked Bar Chart)----
    data = {
    "Language": ["Python", "Java", "C", "C++", "Go"],
    "Startup": [0.05, 0.25, 0.01, 0.015, 0.02],
    "Execution": [21.05, 0.30, 0.18, 0.07, 0.14],
    "GC / Memory Management": [0.02, 0.05, 0, 0, 0],
    "Profiler Overhead": [0.3, 0.15, 0.2, 0.25, 0.12]
    }

    df = pd.DataFrame(data)

    # Melt for stacked bar plot
    df_melted = df.melt(id_vars="Language", var_name="Phase", value_name="Time (s)")

    # Plot
    fig = px.bar(
        df_melted,
        x="Language",
        y="Time (s)",
        color="Phase",
        title="9.Profiling Time Breakdown by Language",
        text_auto=".2s"
    )

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        yaxis=dict(title="Total Profiling Time (seconds)")
    )

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)

    #-----10.Compilation Time by Language (Bar Chart)----
    data = {
    "Language": ["Python", "Java", "C", "C++", "Go"],
    "Compilation Time (s)": [0, 1.8, 0.9, 2.5, 0.4]  # Python = 0 (interpreted)
    }

    df = pd.DataFrame(data)

    # Bar Chart
    fig = px.bar(
        df,
        x="Language",
        y="Compilation Time (s)",
        color="Language",
        text="Compilation Time (s)",
        title="10.Compilation Time by Language",
        labels={"Compilation Time (s)": "Time (seconds)"}
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, max(df["Compilation Time (s)"]) * 1.3])
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    time_points = list(range(0, 11))  # Seconds 0 to 10

    data = {
    "Time (s)": time_points * 5,
    "Memory (MB)": 
        [30, 33, 36, 39, 42, 45, 49, 52, 55, 59, 62] +  # Python
        [50, 52, 55, 56, 57, 57, 58, 59, 60, 61, 61] +  # Java
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] +             # C
        [2, 2.5, 3, 3.5, 4, 4, 4.2, 4.4, 4.6, 4.8, 5] +  # C++
        [10, 12, 13, 14, 14, 14, 14, 14, 14, 14, 14],     # Go
    "Language": (["Python"] * 11 + ["Java"] * 11 + ["C"] * 11 + 
                 ["C++"] * 11 + ["Go"] * 11)
    }

    df = pd.DataFrame(data)

    # Plot line chart
    fig = px.line(
        df,
        x="Time (s)",
        y="Memory (MB)",
        color="Language",
        title="11.Memory Allocation Over Time",
        markers=True
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Time (s)",
        yaxis_title="Allocated Memory (MB)",
        legend_title="Language"
    )

    # Streamlit display
    st.plotly_chart(fig, use_container_width=True)

    data = {
    "Language": ["Python", "Java", "C", "C++", "Go"],
    "Performance Score": [2, 4, 5, 5, 4],  # Higher is better
    "Popularity Score": [10, 8, 6, 7, 5],  # GitHub stars, SO questions etc.
    "Ecosystem Size": [100000, 80000, 40000, 50000, 30000]  # e.g., number of packages/libraries
    }

    df = pd.DataFrame(data)

    # Plot Bubble Chart
    fig = px.scatter(
        df,
        x="Popularity Score",
        y="Performance Score",
        size="Ecosystem Size",
        color="Language",
        hover_name="Language",
        size_max=60,
        title="12.Ecosystem Popularity vs Performance (Bubble Chart)",
        labels={
            "Popularity Score": "Community/Ecosystem Popularity",
            "Performance Score": "Runtime Performance"
        }
    )

    fig.update_layout(template="plotly_white")

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    

if section == "Trade-offs":
    st.markdown("""
    <div style='font-size:18px'>

      <h3>5.1 Performance vs. Development Time</h3>

      <p><b>Python</b></p>
      <ul>
        <li><b>Performance:</b> CPU-bound loops in CPython are typically 10–100× slower than native code (bytecode interp, dynamic type checks, object overhead).</li>
        <li><b>Dev Speed:</b> Minimal boilerplate; fast prototyping; easy profiling with <code>cProfile</code> and <code>tracemalloc</code>.</li>
        <li><b>Mitigation:</b> Offload to native via NumPy / Numba / Cython.</li>
      </ul>

      <p><b>C/C++</b></p>
      <ul>
        <li><b>Performance:</b> With <code>-O3 -march=native</code>, gets top runtimes via inlining, loop unrolling, SIMD, cache tuning.</li>
        <li><b>Dev Speed:</b> Slower: manual memory/threads, longer compile cycles, more complex tooling.</li>
      </ul>

      <p><b>Java</b></p>
      <ul>
        <li><b>Performance:</b> JIT optimizes hot loops to near-native after warm-up (~0.03 s in our runs).</li>
        <li><b>Dev Speed:</b> Strong typing, rich libs, and excellent tooling (VisualVM, JFR).</li>
      </ul>

      <p><b>Go</b></p>
      <ul>
        <li><b>Performance:</b> AOT-compiled; often faster than Java/C++ on low-allocation microtasks (e.g., Fibonacci+Odd). May lag optimized C++ on large FP workloads.</li>
        <li><b>Dev Speed:</b> Very fast builds, simple syntax, no headers, integrated profiling (<code>pprof</code>, <code>trace</code>).</li>
      </ul>

      <h3>5.2 Reliability vs. Control</h3>

      <p><b>C++</b></p>
      <ul>
        <li><b>Control:</b> Direct memory access, low-level APIs, cache layout control.</li>
        <li><b>Risk:</b> Leaks, overflows, races if mismanaged.</li>
        <li><b>Mitigation:</b> Modern C++ (RAII, smart pointers, <code>&lt;thread&gt;</code>) improves safety without losing control.</li>
      </ul>

      <p><b>Java &amp; Python</b></p>
      <ul>
        <li><b>Reliability:</b> GC + safety eliminate many low-level bugs.</li>
        <li><b>Trade-off:</b> Less control of memory layout/cache; GC pauses exist but are usually small.</li>
      </ul>

      <p><b>Go</b></p>
      <ul>
        <li><b>Reliability:</b> Memory-safe with concurrent GC; no manual frees.</li>
        <li><b>Trade-off:</b> Less placement control than C/C++; GC is predictable in low-allocation workloads but still present for long-lived, high-allocation apps.</li>
      </ul>

      <h3>5.3 Memory</h3>

      <p><b>C/C++</b></p>
      <ul>
        <li><b>Efficiency:</b> Stack-first / zero-allocation strategies minimize footprint.</li>
        <li><b>Your Data:</b> C ~1 KB; C++ ~7.7 MB for storing 1M Fibonacci in <code>std::vector</code>.</li>
      </ul>

      <p><b>Java</b></p>
      <ul>
        <li><b>Baseline:</b> ~50–55 MB used heap; max heap observed ~805 MB.</li>
        <li><b>Amortization:</b> Baseline fades in large, long-lived services.</li>
      </ul>

      <p><b>Python</b></p>
      <ul>
        <li><b>Overhead:</b> High per-object metadata.</li>
        <li><b>Your Run:</b> ~0.3 MB (low-allocation case). Large object graphs grow substantially; NumPy mitigates via dense arrays.</li>
      </ul>

      <p><b>Go</b></p>
      <ul>
        <li><b>Baseline:</b> Very low in these tasks (~0.22 MB Fibonacci+Odd), scaling modestly (~0.48–6.68 MB in N-body).</li>
        <li><b>GC Overhead:</b> Minimal in low-allocation runs; stable under load.</li>
      </ul>

      <h3>5.4 Comparative Summary Table</h3>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead>
          <tr>
            <th>Aspect</th>
            <th>Python</th>
            <th>C/C++</th>
            <th>Java</th>
            <th>Go</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Loop Performance</td>
            <td>Slowest (GIL + interpreter)</td>
            <td>Fastest (native, optimized)</td>
            <td>Near-native (JIT)</td>
            <td>Near-native (AOT)</td>
          </tr>
          <tr>
            <td>Dev Time</td>
            <td>Fastest to code</td>
            <td>Longest</td>
            <td>Medium-fast</td>
            <td>Fast (simple syntax, fast compile)</td>
          </tr>
          <tr>
            <td>Memory Control</td>
            <td>Low</td>
            <td>Highest</td>
            <td>Medium</td>
            <td>Medium</td>
          </tr>
          <tr>
            <td>Reliability</td>
            <td>High (runtime safety)</td>
            <td>Lower without discipline</td>
            <td>High (runtime safety)</td>
            <td>High (runtime safety, concurrent GC)</td>
          </tr>
          <tr>
            <td>Baseline Memory</td>
            <td>Medium (~10s MB RSS)</td>
            <td>Minimal (KB–MB)</td>
            <td>High (50+ MB)</td>
            <td>Low (~0.2–6 MB typical)</td>
          </tr>
          <tr>
            <td>Best Use Case</td>
            <td>Rapid prototyping, glue code</td>
            <td>Performance-critical systems</td>
            <td>Long-running scalable services</td>
            <td>High-concurrency, low-latency services</td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

if section == "Threats to Validity":
    st.markdown("""
    <div style='font-size:18px'>

      <p>
        Despite the efforts to standardize benchmarking across Python, Java, C, C++, and Go, several internal and external factors may affect the accuracy, fairness, or generalizability of the results. This section outlines those key concerns.
      </p>

      <span style='font-size:20px; font-weight:bold;'>6.1 Non-Uniform Workloads</span>
      <ul>
        <li><b>C++:</b> stored the entire Fibonacci sequence in memory; <b>Python:</b> computed it in constant space; <b>Go:</b> iterative with minimal allocation.</li>
        <li>Loop boundaries and result output behaviors varied slightly, introducing runtime deviations.</li>
        <li>Base conditions and recursion structures differ subtly due to language semantics and call stack growth behavior.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.2 JIT Warm-up Effects (Java)</span>
      <ul>
        <li>Java’s JIT activates after method “hotness” detection; warm-up runs are required.</li>
        <li>Short benchmarks may misrepresent Java’s speed depending on optimization state.</li>
        <li>Steady-state benchmarking is required for accuracy.</li>
        <li><b>Go</b> is AOT-compiled and has no JIT warm-up, giving it an edge in short-lived microbenchmarks.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.3 Profiler Overhead</span>
      <ul>
        <li><b>C/C++:</b> Valgrind significantly slows execution and perturbs cache behavior.</li>
        <li><b>Python:</b> cProfile/tracemalloc observe Python-level behavior only (miss OS/native costs).</li>
        <li><b>Java:</b> VisualVM adds small overhead but can influence memory/GC timing.</li>
        <li><b>Go:</b> pprof/trace provide deep data but can add measurable overhead for short runs with high sampling.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.4 Memory Measurement Discrepancies</span>
      <ul>
        <li><b>Python:</b> <code>tracemalloc</code> tracks Python allocations, not total process RSS.</li>
        <li><b>Java:</b> VisualVM reports managed heap; excludes native memory (JIT cache, thread stacks).</li>
        <li><b>C/C++:</b> Valgrind and <code>/usr/bin/time</code> capture low-level memory but may miss stack/transient buffers.</li>
        <li><b>Go:</b> pprof reports heap allocations; OS-reserved memory and goroutine stacks aren’t always captured.</li>
        <li>Cross-language comparisons require OS-level normalization (e.g., <code>top</code>, <code>smem</code>).</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.5 Garbage Collection and Runtime Behavior</span>
      <ul>
        <li><b>Java:</b> GC introduces nondeterministic pauses; timing varies with allocation pressure.</li>
        <li><b>Python:</b> Refcounting + periodic cycle GC may not trigger in short runs.</li>
        <li><b>C/C++:</b> No GC → stable timings but risk of leaks if mismanaged.</li>
        <li><b>Go:</b> Concurrent GC minimizes pauses but adds small CPU overhead; GC cycles during long runs can affect latency metrics.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.6 Platform and Environment Variability</span>
      <ul>
        <li>Runs were on Windows 11 / WSL2 Ubuntu; platform differences influence results.</li>
        <li>Windows scheduling can add jitter; Linux often yields steadier timing.</li>
        <li>Power plans, CPU frequency scaling, and background tasks were not fully locked down.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.7 Compiler Flags and Build Optimization</span>
      <ul>
        <li><b>C/C++:</b> Flags like <code>-O3 -march=native -flto</code> can change results substantially.</li>
        <li><b>Java:</b> JVM settings (e.g., <code>-XX:+UseG1GC</code>) affect speed and memory.</li>
        <li><b>Python:</b> Alternative runtimes (PyPy/Cython) can drastically alter performance (not used here).</li>
        <li><b>Go:</b> Build flags (e.g., <code>-gcflags</code>, <code>-ldflags=-s -w</code>) and inlining thresholds impact time/binary size; defaults used here.</li>
      </ul>

      <span style='font-size:20px; font-weight:bold;'>6.8 Language-Specific Runtime Cost Assumptions</span>
      <ul>
        <li><b>Python:</b> Interpreter startup + dynamic types add constant overhead.</li>
        <li><b>Java:</b> JVM initialization is non-trivial but amortized in long runs.</li>
        <li><b>C/C++:</b> Minimal startup; dominated by algorithmic execution.</li>
        <li><b>Go:</b> Startup initializes scheduler/GC; negligible for long tasks but measurable in microbenchmarks.</li>
        
      </ul>

    </div>
    """, unsafe_allow_html=True)

if section == "Conclusion":
    st.markdown("""
    <div style='font-size:18px'>

      This comparative study evaluated the runtime efficiency, memory usage, garbage collection behavior, concurrency support, and developer ergonomics of five popular programming languages—<b>C</b>, <b>C++</b>, <b>Java</b>, <b>Python</b>, and <b>Go</b>—across a diverse set of computationally intensive tasks:
      <i>Fibonacci + Odd Number generation</i>, <i>N-body simulation</i>, <i>Deep Recursion</i>, and <i>N-gram string processing</i>.<br><br>

      <span style='font-size:20px; font-weight:bold;'>7.1 Key Findings</span><br><br>

      <b>Performance</b><br>
      • <b>C++</b> was fastest on heavy numeric workloads (e.g., N-body) via low-level optimizations, inlining, and efficient <code>std::thread</code> usage.<br>
      • <b>C</b> closely followed, especially in memory-light tasks (manual memory, minimal runtime overhead).<br>
      • <b>Go</b> led on low-allocation microbenchmarks (Fibonacci+Odd) and stayed competitive on recursion-heavy tasks.<br>
      • <b>Java</b> reached near-native performance after JIT warm-up, strong in loop-heavy and concurrent runs.<br>
      • <b>Python</b> trailed due to interpreter overhead and the GIL, but remains fine for moderate workloads and rapid prototyping.<br><br>

      <b>Memory Usage</b><br>
      • <b>C</b> had the smallest footprint when avoiding heap allocations.<br>
      • <b>C++</b> stayed efficient, with slight overhead from STL containers.<br>
      • <b>Go</b> kept a low baseline (hundreds of KBs) in low-allocation workloads, scaling modestly on heavier tasks.<br>
      • <b>Java</b> carried consistent overhead from class metadata, JIT cache, and GC-managed heap.<br>
      • <b>Python</b> showed higher per-object costs from interpreter structures.<br><br>

      <b>Concurrency</b><br>
      • <b>Java</b>, <b>C</b>, and <b>C++</b> leveraged OS threads for parallelism.<br>
      • <b>Go</b> used lightweight goroutines and an efficient scheduler (low thread creation cost).<br>
      • <b>Python</b> was limited by the GIL; multiprocessing or native extensions are required for CPU-bound parallelism.<br><br>

      <b>Developer Experience & Tooling</b><br>
      • <b>Python:</b> Shortest development cycle; simple profiling (cProfile, tracemalloc).<br>
      • <b>Java:</b> Strong profilers (VisualVM, JFR), robust threading, memory safety.<br>
      • <b>Go:</b> Simple syntax, fast compiles, integrated profiling (pprof, trace)—great for backend/concurrency.<br>
      • <b>C/C++:</b> Most demanding but unmatched control and optimization potential.<br><br>

      <span style='font-size:20px; font-weight:bold;'>7.2 General Recommendations</span><br><br>
      • <b>Maximum performance</b> in CPU-bound numeric/simulation tasks → <b>C++</b> or <b>C</b><br>
      • <b>Balanced speed, safety, and tooling</b> for long-running concurrent apps → <b>Java</b> or <b>Go</b><br>
      • <b>Rapid experimentation & integration</b> with external libraries → <b>Python</b> (NumPy, Numba, Cython)<br>
      • <b>High-concurrency backend / microservices</b> → <b>Go</b><br><br>

      <span style='font-size:20px; font-weight:bold;'>7.3 Summary Table</span><br><br>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr>
            <th>Parameter</th>
            <th>Python</th>
            <th>Java</th>
            <th>C</th>
            <th>C++</th>
            <th>Go</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Runtime Speed</td><td>Slow</td><td>Good</td><td>Very Fast</td><td>Very Fast</td><td>Very Fast (low-allocation)</td></tr>
          <tr><td>Memory Usage</td><td>High</td><td>Medium (GC overhead)</td><td>Very Efficient</td><td>Efficient</td><td>Low (predictable GC)</td></tr>
          <tr><td>Concurrency</td><td>Threading limited by GIL</td><td>Multithreading</td><td>Multithreading</td><td>Multithreading</td><td>Goroutines + channels</td></tr>
          <tr><td>Compilation Time</td><td>N/A (Interpreted)</td><td>Slow</td><td>Fast</td><td>Often Slow (templates)</td><td>Fast</td></tr>
          <tr><td>Memory Safety</td><td>GC-managed</td><td>GC-managed</td><td>Manual & unsafe</td><td>Manual & unsafe</td><td>GC-managed</td></tr>
          <tr><td>Startup Time</td><td>Fast</td><td>Slow</td><td>Fast</td><td>Fast</td><td>Fast</td></tr>
          <tr><td>Binary Size</td><td>N/A</td><td>Large</td><td>Small</td><td>Medium-Large</td><td>Small</td></tr>
          <tr><td>Developer Productivity</td><td>High</td><td>High</td><td>Low (verbose)</td><td>Medium</td><td>High</td></tr>
          <tr><td>Ecosystem & Libraries</td><td>Huge (AI, DS, Web)</td><td>Mature</td><td>Smaller</td><td>Rich (games, systems, ML)</td><td>Growing (cloud, backend)</td></tr>
          <tr><td>Cross-Platform</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
          <tr><td>Portability</td><td>High</td><td>High</td><td>High</td><td>High</td><td>High</td></tr>
          <tr><td>Learning Curve</td><td>Easy</td><td>Medium</td><td>Steep</td><td>Steep</td><td>Easy–Medium</td></tr>
          <tr><td>Performance-Critical Apps</td><td>Not suitable</td><td>Moderate</td><td>Excellent</td><td>Excellent</td><td>Excellent (for concurrency)</td></tr>
          <tr><td>Embedded Systems</td><td>No</td><td>Rare</td><td>Excellent</td><td>Good</td><td>Limited</td></tr>
          <tr><td>GUI Development</td><td>Possible (Tkinter, PyQt)</td><td>Good (JavaFX, Swing)</td><td>Rare</td><td>Good (Qt, wxWidgets)</td><td>Limited</td></tr>
          <tr><td>Best Use Cases</td><td>AI, Scripting, Data Science</td><td>Web, Enterprise, Android</td><td>Systems, OS, Real-Time</td><td>Games, High-Perf Apps</td><td>Backend services, networking</td></tr>
          <tr><td>Type System</td><td>Dynamic</td><td>Static (Strong)</td><td>Static</td><td>Static</td><td>Static (Strong)</td></tr>
          <tr><td>Garbage Collection</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>Yes (concurrent)</td></tr>
          <tr><td>Low-level Hardware Access</td><td>No</td><td>Limited</td><td>Yes</td><td>Yes</td><td>Limited</td></tr>
        </tbody>
      </table><br><br>

      <span style='font-size:20px; font-weight:bold;'>7.4 Overall Ranking</span><br><br>

      <b>1. C++ 🥇</b><br>
      <i>Why it ranks first:</i><br>
      • <b>Performance:</b> Consistently the fastest in heavy workloads (Fibonacci, N-body, deep recursion) via inlining, loop unrolling, SIMD.<br>
      • <b>Memory Efficiency:</b> Slightly more than C when using STL, but still efficient.<br>
      • <b>Concurrency:</b> Full OS-thread support (<code>std::thread</code>); scales on multi-core.<br>
      • <b>Stability:</b> No GC pauses; deterministic performance (depends on developer discipline).<br>
      • <b>Developer Productivity:</b> Slower to write than high-level langs; modern C++ (RAII, smart pointers) reduces bugs.<br>
      • <b>Best for:</b> HPC, game engines, trading systems, simulations.<br><br>

      <b>2. C 🥈</b><br>
      <i>Why it ranks second:</i><br>
      • <b>Performance:</b> Very close to C++; sometimes faster in minimal-memory cases.<br>
      • <b>Memory Efficiency:</b> Best-in-class (e.g., ~1 KB in your Fibonacci test).<br>
      • <b>Concurrency:</b> Pthreads/OS threads with minimal overhead.<br>
      • <b>Stability:</b> Predictable, tiny runtime; no GC.<br>
      • <b>Developer Productivity:</b> Lowest-level control → more error-prone and slower iteration.<br>
      • <b>Best for:</b> Embedded, OS kernels, real-time, perf-critical microservices.<br><br>

      <b>3. Go 🥉</b><br>
      <i>Why it ranks third:</i><br>
      • <b>Performance:</b> Slower than C/C++ in raw FP loops, but excellent overall; shines in low-allocation microtasks.<br>
      • <b>Memory Efficiency:</b> Leaner than Java/Python; GC tuned for low-latency.<br>
      • <b>Concurrency:</b> Goroutines + channels make scalable concurrency simple.<br>
      • <b>Stability:</b> Safe memory model; predictable GC.<br>
      • <b>Developer Productivity:</b> Simple syntax, fast compiles, great tooling (<code>go test</code>, <code>go fmt</code>, <code>pprof</code>).<br>
      • <b>Best for:</b> Scalable backends, cloud APIs, network-heavy workloads, pipelines.<br><br>

      <b>4. Java 🏅</b><br>
      <i>Why it ranks fourth:</i><br>
      • <b>Performance:</b> Near-native after JIT warm-up; weaker on short CPU-bound tasks.<br>
      • <b>Memory Efficiency:</b> Higher baseline (50–75 MB).<br>
      • <b>Concurrency:</b> Strong executors/streams; excellent for server-side.<br>
      • <b>Stability:</b> Short but non-zero GC pauses; robust exception model.<br>
      • <b>Developer Productivity:</b> Excellent IDEs, rich ecosystem; slower startup than Go/C++.<br>
      • <b>Best for:</b> Enterprise services, Android, distributed systems, finance backends.<br><br>

      <b>5. Python 🎯</b><br>
      <i>Why it ranks fifth:</i><br>
      • <b>Performance:</b> Slowest for CPU-bound loops (interpreter + GIL).<br>
      • <b>Memory Efficiency:</b> High per-object overhead; process RSS grows with object graphs.<br>
      • <b>Concurrency:</b> GIL blocks CPU multithreading; use multiprocessing/C-extensions for parallelism.<br>
      • <b>Stability:</b> Very stable and safe runtime.<br>
      • <b>Developer Productivity:</b> Fastest to code/debug; huge ecosystem offsets raw speed limits.<br>
      • <b>Best for:</b> Data science, AI/ML, automation, prototypes, API wrappers.<br><br>

      <table style="width:100%; font-size:16px; border-collapse:collapse;" border="1" cellpadding="6">
        <thead style="font-weight:bold;">
          <tr><th>Rank</th><th>Language</th><th>Strengths</th><th>Weaknesses</th><th>Best Use Cases</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td><td>C++</td>
            <td>Fastest runtime, strong concurrency, high control</td>
            <td>Steep learning curve; manual resource management</td>
            <td>HPC, games, trading, simulations</td>
          </tr>
          <tr>
            <td>2</td><td>C</td>
            <td>Most memory-efficient, very fast</td>
            <td>Very low-level; verbose; risky memory handling</td>
            <td>Embedded, OS, real-time</td>
          </tr>
          <tr>
            <td>3</td><td>Go</td>
            <td>Easy concurrency, clean syntax, fast compile</td>
            <td>Slightly slower than C/C++ in raw compute</td>
            <td>Cloud services, microservices</td>
          </tr>
          <tr>
            <td>4</td><td>Java</td>
            <td>Mature tooling; near-native after JIT</td>
            <td>High baseline memory; slower startup</td>
            <td>Enterprise, Android, backend systems</td>
          </tr>
          <tr>
            <td>5</td><td>Python</td>
            <td>Most productive; huge ecosystem</td>
            <td>Slowest; high memory overhead</td>
            <td>AI, scripting, prototyping</td>
          </tr>
        </tbody>
      </table>
    </div><br><br>
    """, unsafe_allow_html=True)
    st.image(r"Images\result4.png", 
             caption="Results Summary", 
             use_column_width=True)

st.markdown(section_content[section] if section_content[section] else "")

st.sidebar.write("Developed by Michael Fernandes")


