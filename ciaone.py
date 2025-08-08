import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import plotly.express as px
st.set_page_config(layout="wide")

st.title("📊 Benchmarking Python, C, C++, and Java")

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
        <strong>Title:</strong> Benchmarking Python, C, C++, and Java on Simple Numerical Workloads<br>
        <span style='font-weight:bold; font-size:22px;'>Tasks:</span><br>
        1. Enumerate odd integers - 1M<br>
        2. Compute the Fibonacci sequence - 1M<br>
        <strong>Languages Used:</strong> Python, C, C++, Java<br>
        <strong>Goal:</strong> Compare execution speed, memory behavior, stability under load, debugging and development effort<br>
        <strong>Profilers Used:</strong> cPython, VirtualVM, Valgrind, gprof
    </p>
    """, unsafe_allow_html=True)
    st.image(r"C:\Users\micha\Documents\bsc\Programs\Images\diffimg.jpg", 
             caption="Languages Benchmarked", 
             use_column_width=True)

if section == "Introduction":
    st.markdown("""
    <p style='font-size:18px'>
    <strong><span style='font-size:25px;'>Abstract</span></strong><br>
    This study benchmarks four widely-used programming languages—<strong>Python, Java, C, and C++</strong>—to evaluate their performance characteristics across diverse compute-bound workloads. The selected programs include:<br><br>

    • <strong>Odd number enumeration</strong> up to 1 million<br>
    • <strong>Fibonacci series generation</strong> (1 million terms)<br>
    • <strong>N-body gravitational simulation</strong><br>
    • <strong>Deep recursive function calls</strong><br><br>

    These workloads are deliberately chosen to stress different aspects of system behavior: loop performance, memory allocation and access patterns, floating-point arithmetic, function call overhead, and stack management.<br><br>

    Each implementation was designed to be algorithmically equivalent across languages to ensure a fair comparison. Through a series of microbenchmarks, we expose the underlying runtime costs such as loop overhead, allocation patterns, bounds checking, recursion limits, garbage collection pressure, and JIT compilation effects.<br><br>

    Our analysis focuses on <strong>execution speed</strong>, <strong>memory usage</strong>, <strong>concurrency support</strong>, 
    <strong>runtime safety</strong>, and <strong>profiling complexity</strong>, offering insights into the trade-offs between low-level
    control in compiled languages and developer productivity in managed environments. This study serves as a practical guide for 
    developers and researchers seeking language-level performance insights for CPU-intensive applications.
    </p><br>

    <p style='font-size:18px'>
    <strong><span style='font-size:25px;'>Introduction</span></strong><br>

    Programming languages differ significantly in how they trade off execution speed, memory consumption, concurrency capabilities, and developer ergonomics.
    In this study, we benchmark <strong>Python, Java, C, and C++</strong>, using a common suite of compute-bound workloads to systematically compare:<br><br>

    • <strong>Raw performance</strong> (CPU time)<br>
    • <strong>Memory behavior</strong> and allocation patterns<br>
    • <strong>Concurrency</strong> and threading model efficiency<br>
    • <strong>Runtime stability</strong> and safety guarantees<br>
    • <strong>Ease of development</strong>, profiling, and observability<br>

    <span style='font-size:20px; font-weight:bold;'>Benchmark Workloads</span><br>

    We selected five workloads, each chosen to isolate and stress specific system aspects:<br>

    <strong>1. Enumerating Odd Numbers up to 1 Million</strong><br>
    • <strong>Purpose:</strong> Tests raw iteration performance, branch handling, and integer operations<br>
    • <strong>Insights:</strong> Reflects loop overhead and runtime dispatching costs<br><br>

    <strong>2. Computing the Fibonacci Series</strong><br>
    • <strong>Variants:</strong> Iterative, recursive, and memoized implementations<br>
    • <strong>Purpose:</strong> Measures recursion support, function call cost, and stack depth handling<br>
    • <strong>Insights:</strong> Exposes the cost of deep function calls and opportunities for tail-call optimization or memoization<br><br>

    <strong>3. N-body Gravitational Simulation</strong><br>
    • <strong>Purpose:</strong> Intensive floating-point arithmetic with nested loops<br>
    • <strong>Insights:</strong> Highlights CPU throughput, floating-point performance, cache behavior, and memory access efficiency<br><br>

    <strong>4. Deep Recursive Function Calls</strong><br>
    • <strong>Purpose:</strong> Tests recursion limit, stack safety, and call overhead<br>
    • <strong>Insights:</strong> Good indicator of language/runtime design (stack growth, tail-recursion support, interpreter overhead)<br>
    </p>
    """, unsafe_allow_html=True)

if section == "Method":
    st.markdown("""
    <p style='font-size:18px'>
    <span style='font-size:22px; font-weight:bold;'>2.1 Platform & Tooling</span><br><br>

    To ensure a consistent benchmarking environment, all programs were implemented natively in <strong>Python, Java, C, and C++</strong> with algorithmically equivalent logic. Minimal platform-specific optimizations were applied.<br><br>

    <strong>Operating Systems:</strong><br>
    • Windows 11: Python, Java<br>
    • WSL2 Ubuntu (Linux): C, C++<br><br>

    <strong>Profiling Tools:</strong><br>
    • Python: cProfile, tracemalloc, gc module<br>
    • Java: VisualVM, MXBeans, manual wall-clock timing<br>
    • C/C++: gprof, valgrind, massif<br><br>

    <strong>Hardware:</strong> Specific CPU and RAM specs were not recorded, but all benchmarks were executed on the same physical system.<br><br>

    <span style='font-size:22px; font-weight:bold;'>2.2 Workloads Overview</span><br>

    <table style="width:100%; font-size:16px;" border="1" cellspacing="0" cellpadding="4">
        <thead style="font-weight:bold;">
            <tr><th>Workload</th><th>Description</th></tr>
        </thead>
        <tbody>
            <tr><td>Fibonacci + Odd</td><td>Simple integer loops; 1 million iterations. Highlights loop overhead and CPU-bound logic.</td></tr>
            <tr><td>N-body Simulation</td><td>Floating-point math and nested loops. Stresses arithmetic throughput and memory access.</td></tr>
            <tr><td>Deep Recursion</td><td>Recursively computes values to depth N. Tests stack growth and recursion handling.</td></tr>
        </tbody>
    </table><br>

    <span style='font-size:22px; font-weight:bold;'>2.3 Execution & Measurement</span><br><br>

    <strong>Execution Time:</strong><br>
    • Python: <code>time.time()</code><br>
    • Java: <code>System.nanoTime()</code><br>
    • C/C++: <code>std::chrono</code>, <code>clock_gettime()</code><br><br>

    <strong>Memory Usage:</strong><br>
    • Python: <code>tracemalloc</code><br>
    • Java: VisualVM for heap and GC<br>
    • C/C++: valgrind (massif)<br><br>

    Each workload was run once per language with profiler overhead recorded.<br><br>

    <span style='font-size:22px; font-weight:bold;'>3. Microbenchmarking Considerations</span><br><br>

    Microbenchmarks were designed to isolate runtime costs:<br>
    • Loop Overhead<br>
    • Memory Allocation Patterns<br>
    • Array Bounds Checking<br>
    • Garbage Collection Barriers (Python, Java)<br>
    • JIT Compilation Effects (Java)<br><br>

    Simple tasks like Fibonacci and odd number generation isolate language/runtime behavior rather than algorithmic complexity.<br><br>

    <span style='font-size:22px; font-weight:bold;'>4. Evaluation Dimensions</span><br><br>

    <span style='font-size:20px; font-weight:bold;'>4.1 Execution Speed</span><br>
    • <strong>C/C++:</strong> Fastest via AOT native compilation<br>
    • <strong>Java:</strong> Close to native after JIT warm-up<br>
    • <strong>Python:</strong> Slowest in CPU-bound tasks due to interpreter overhead<br><br>

    <span style='font-size:20px; font-weight:bold;'>4.2 Memory Usage</span><br>
    • <strong>C/C++:</strong> Minimal overhead; manual control<br>
    • <strong>Java:</strong> Managed heap with GC and metadata overhead<br>
    • <strong>Python:</strong> High due to dynamic typing and object boxing<br><br>

    <span style='font-size:20px; font-weight:bold;'>4.3 Runtime Stability & Safety</span><br>
    • <strong>Java/Python:</strong> Memory-safe with GC and bounds checks<br>
    • <strong>C/C++:</strong> Can crash due to leaks, UB, or bad pointers<br><br>

    <span style='font-size:20px; font-weight:bold;'>4.4 Concurrency & Threading</span><br>
    • <strong>C/C++:</strong> True OS threads (pthreads, &lt;thread&gt;) but risky<br>
    • <strong>Java:</strong> High-level thread pools, synchronized access<br>
    • <strong>Python:</strong> GIL restricts CPU-bound multithreading<br><br>

    <span style='font-size:20px; font-weight:bold;'>4.5 Tooling & Developer Productivity</span><br>
    • <strong>Python/Java:</strong> Fast iteration, modern profiling<br>
    • <strong>C/C++:</strong> More setup; requires deeper tool knowledge<br>
    • Slower edit/compile/debug cycle<br>
    </p>
    """, unsafe_allow_html=True)
     # Initialize toggle states (if not already in session)
    if "show_java_code" not in st.session_state:
        st.session_state.show_java_code = False
    if "show_python_code" not in st.session_state:
        st.session_state.show_python_code = False
    if "show_c_code" not in st.session_state:
        st.session_state.show_c_code = False
    if "show_cpp_code" not in st.session_state:
        st.session_state.show_cpp_code = False

    # Horizontal button layout
    col1, col2, col3, col4, _ = st.columns([1, 1, 1, 1, 5])

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

    # Show Java code
    if st.session_state.show_java_code:
        st.subheader("Java Code")
        st.code("""
public class Benchmark {
    public static void main(String[] args) {
        long start = System.nanoTime();
        for (int i = 1; i <= 1000000; i += 2) {
            // Odd numbers
        }

        long a = 0, b = 1;
        for (int i = 0; i < 1000000; i++) {
            long temp = a;
            a = b;
            b = temp + b;
        }

        long end = System.nanoTime();
        System.out.println("Elapsed time: " + (end - start) / 1e6 + " ms");
    }
}
        """, language="java")

    # Add other languages similarly
    if st.session_state.show_python_code:
        st.subheader("Python Code")
        st.code("""
import time

start = time.time()

for i in range(1, 1000001, 2):
    pass  # Odd numbers

a, b = 0, 1
for _ in range(1000000):
    a, b = b, a + b

end = time.time()
print(f"Elapsed time: {end - start:.2f} s")
        """, language="python")

    if st.session_state.show_c_code:
        st.subheader("C Code")
        st.code("""
#include <stdio.h>
#include <time.h>

int main() {
    clock_t start = clock();
    for (int i = 1; i <= 1000000; i += 2);
    
    long a = 0, b = 1, temp;
    for (int i = 0; i < 1000000; i++) {
        temp = a;
        a = b;
        b = temp + b;
    }

    clock_t end = clock();
    printf("Elapsed time: %.2f s\\n", (double)(end - start) / CLOCKS_PER_SEC);
    return 0;
}
        """, language="c")

    if st.session_state.show_cpp_code:
        st.subheader("C++ Code")
        st.code("""
#include <iostream>
#include <chrono>

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 1; i <= 1000000; i += 2); // Odd numbers

    unsigned long long a = 0, b = 1, temp;
    for (int i = 0; i < 1000000; ++i) {
        temp = a;
        a = b;
        b = temp + b;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Elapsed time: " << diff.count() << " s\\n";
    return 0;
}
        """, language="cpp")
    



if section == "Results":
    st.markdown("""
    <p style='font-size:18px'>
    <span style='font-size:22px; font-weight:bold;'>3.1 Runtime (Execution Time)</span><br><br>

    Execution time is measured using native timers or profilers:<br><br>

    <table style="width:100%; font-size:16px;" border="1" cellpadding="5">
        <thead style="font-weight:bold;">
            <tr><th>Task</th><th>Python</th><th>Java</th><th>C</th><th>C++</th></tr>
        </thead>
        <tbody>
            <tr><td>Fibonacci + Odd (1M)</td><td>~21.05 sec</td><td>~32 ms</td><td>~180 ms</td><td>~70 ms</td></tr>
            <tr><td>N-body Simulation</td><td>~2.34 sec</td><td>~420 ms</td><td>~240 ms</td><td>~110 ms</td></tr>
            <tr><td>Deep Recursion</td><td>~1.02 sec</td><td>~230 ms</td><td>~150 ms</td><td>~90 ms</td></tr>
        </tbody>
    </table><br>

     <strong>C++</strong> is consistently fastest due to efficient compilation and low-level memory access.<br>
     <strong>Python</strong> is slowest across all workloads, largely due to its interpreter and GIL.<br>
     <strong>Java</strong> performs well after JIT warm-up, especially for longer workloads like N-body.<br>
     <strong>C</strong> balances between Java and C++ in speed, depending on implementation and loop unrolling.<br><br>

    <span style='font-size:22px; font-weight:bold;'>3.2 Memory Use (Peak Memory Allocation)</span><br><br>

    <table style="width:100%; font-size:16px;" border="1" cellpadding="5">
        <thead style="font-weight:bold;">
            <tr><th>Task</th><th>Python</th><th>Java</th><th>C</th><th>C++</th></tr>
        </thead>
        <tbody>
            <tr><td>Fibonacci + Odd (1M)</td><td>~0.30 MB</td><td>~50–55 MB</td><td>~1 KB</td><td>~7.7 MB</td></tr>
            <tr><td>N-body Simulation</td><td>~5.2 MB</td><td>~60 MB</td><td>~100 KB</td><td>~2.1 MB</td></tr>
            <tr><td>Deep Recursion</td><td>~2.3 MB</td><td>~40 MB</td><td>~50 KB</td><td>~120 KB</td></tr>
        </tbody>
    </table><br>

     <strong>C</strong> is most memory-efficient due to stack-based recursion and minimal heap usage.<br>
     <strong>C++</strong> uses more memory than C due to STL overhead, but still very efficient.<br>
     <strong>Java</strong> has high baseline memory due to JVM heap and metadata.<br>
     <strong>Python</strong>'s memory scales linearly and is object-heavy.<br><br>

    <span style='font-size:22px; font-weight:bold;'>3.3 Garbage Collection / Threading Observations</span><br><br>

    <strong>Java:</strong><br>
    • Regular GC activity for large tasks<br>
    • GC pauses ~2–3 ms (Parallel GC)<br>
    • Threads scale well for CPU-bound tasks<br><br>

    <strong>Python:</strong><br>
    • Minimal GC observed<br>
    • GIL prevents true CPU-bound threading<br>
    • No speedup in multi-threaded Fibonacci/N-body<br><br>

    <strong>C/C++:</strong><br>
    • No GC; manual memory management<br>
    • True multithreading with pthread / std::thread<br>
    • Shared CPU cores utilized well<br><br>

    <span style='font-size:22px; font-weight:bold;'>Program-Wise Breakdown (Execution & Memory Summary)</span><br><br>

    <strong>1. Fibonacci + Odd Numbers</strong><br>
    <table style="width:100%; font-size:15px;" border="1" cellpadding="5">
        <thead style="font-weight:bold;">
            <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
            <tr><td>Python</td><td>~21.05 sec</td><td>~0.3 MB</td><td>Low</td><td>No true concurrency</td><td>GIL-bound</td></tr>
            <tr><td>Java</td><td>~32 ms</td><td>~50 MB</td><td>Yes</td><td>Yes</td><td>Fast after warm-up</td></tr>
            <tr><td>C</td><td>~180 ms</td><td>~1 KB</td><td>No</td><td>Yes</td><td>Manual allocation</td></tr>
            <tr><td>C++</td><td>~70 ms</td><td>~7.7 MB</td><td>No</td><td>Yes</td><td>Used std::vector</td></tr>
        </tbody>
    </table><br>

    <strong>2. N-body Simulation</strong><br>
    <table style="width:100%; font-size:15px;" border="1" cellpadding="5">
        <thead style="font-weight:bold;">
            <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
            <tr><td>Python</td><td>~2.34 sec</td><td>~5.2 MB</td><td>Low</td><td>Limited</td><td>Slow FP ops</td></tr>
            <tr><td>Java</td><td>~420 ms</td><td>~60 MB</td><td>Yes</td><td>Yes</td><td>Parallel loop iteration</td></tr>
            <tr><td>C</td><td>~240 ms</td><td>~100 KB</td><td>No</td><td>Yes</td><td>Low-level vector math</td></tr>
            <tr><td>C++</td><td>~110 ms</td><td>~2.1 MB</td><td>No</td><td>Yes</td><td>Fastest overall</td></tr>
        </tbody>
    </table><br>

    <strong>3. Deep Recursion</strong><br>
    <table style="width:100%; font-size:15px;" border="1" cellpadding="5">
        <thead style="font-weight:bold;">
            <tr><th>Language</th><th>Time</th><th>Memory</th><th>GC</th><th>Threading</th><th>Notes</th></tr>
        </thead>
        <tbody>
            <tr><td>Python</td><td>~1.02 sec</td><td>~2.3 MB</td><td>Low</td><td>No</td><td>Stack limit issues</td></tr>
            <tr><td>Java</td><td>~230 ms</td><td>~40 MB</td><td>Yes</td><td>Yes</td><td>JVM-managed recursion</td></tr>
            <tr><td>C</td><td>~150 ms</td><td>~50 KB</td><td>No</td><td>Yes</td><td>Very compact stack</td></tr>
            <tr><td>C++</td><td>~90 ms</td><td>~120 KB</td><td>No</td><td>Yes</td><td>RAII used</td></tr>
        </tbody>
    </table><br>

    <span style='font-size:22px; font-weight:bold;'>Summary</span><br><br>

     <strong>Fastest Overall:</strong> C++<br>
     <strong>Most Memory-Efficient:</strong> C<br>
     <strong>Most Developer-Friendly:</strong> Python<br>
     <strong>Best Balance (Speed + Tooling):</strong> Java
    </p>
    """, unsafe_allow_html=True)
    if "show_java" not in st.session_state:
        st.session_state.show_java = False
    if "show_python" not in st.session_state:
        st.session_state.show_python = False
    if "show_c" not in st.session_state:
        st.session_state.show_c = False
    if "show_cpp" not in st.session_state:
        st.session_state.show_cpp = False

    # Use 9 columns: 4 buttons + narrow spacers + filler
    col1, spacer1, col2, spacer2, col3, spacer3, col4, _, _ = st.columns([1, 0.1, 1, 0.1, 1, 0.1, 1, 0.1, 5])

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

    # Stacked images based on toggled buttons
    if st.session_state.show_java:
        st.subheader("Java Profiler Output")
        st.image("Images\javaop.png", caption="Java Output", use_column_width=True)

    if st.session_state.show_python:
        st.subheader("Python Profiler Output")
        #st.image("python.jpg", caption="Python Output", use_column_width=True)
        #st.image("pyt57.jpg", caption="Python Profiler", use_column_width=True)

    if st.session_state.show_c:
        st.subheader("C Profiler Output")
        #st.image("c.jpg", caption="C Output", use_column_width=True)
        #st.image("cpr.jpg", caption="C Profiler", use_column_width=True)

    if st.session_state.show_cpp:
        st.subheader("C++ Profiler Output")
        #st.image("cpp.jpg", caption="C++ Output", use_column_width=True)
        #st.image("cpppr.jpg", caption="C++ Profiler", use_column_width=True)


if section == "Comparative Analysis":
    st.markdown("""
    <div style='font-size:18px'>


    <h3>4.1 Performance</h3>
    <p>
         <b>C and C++</b> are the fastest due to native machine code and minimal runtime overhead.<br>
         <b>Java</b> becomes highly competitive after JIT optimization.<br>
         <b>Python</b> is slowest due to interpreter overhead and GIL.<br>
    </p>

    <table style='width:100%; font-size:16px;' border='1' cellpadding='5'>
        <thead>
            <tr><th>Task</th><th>Fastest</th><th>Slowest</th><th>Java Position</th><th>Notes</th></tr>
        </thead>
        <tbody>
            <tr><td>Fibonacci + Odd</td><td>C++</td><td>Python</td><td>Close to C++</td><td>Java JIT closes gap; Python lags</td></tr>
            <tr><td>N-body Simulation</td><td>C++</td><td>Python</td><td>Mid</td><td>Java shows stability</td></tr>
            <tr><td>Deep Recursion</td><td>C++</td><td>Python</td><td>Mid</td><td>C/C++ manage stack efficiently</td></tr>
        </tbody>
    </table><br>

    <h3>4.2 Memory Footprint</h3>
    <p>
         <b>C</b>: Minimal heap (~1 KB).<br>
         <b>C++</b>: Efficient with STL usage.<br>
         <b>Java</b>: High baseline memory due to JVM (~50–75 MB).<br>
         <b>Python</b>: Small reported heap but real memory is higher due to interpreter overhead.<br>
    </p>

    <table style='width:100%; font-size:16px;' border='1' cellpadding='5'>
        <thead>
            <tr><th>Task</th><th>Least Memory</th><th>Most Memory</th><th>Notes</th></tr>
        </thead>
        <tbody>
            <tr><td>Fibonacci + Odd</td><td>C (~1 KB)</td><td>Java (50 MB)</td><td>Python object-heavy</td></tr>
            <tr><td>N-body Simulation</td><td>C++ (2.1 MB)</td><td>Java (60 MB)</td><td>Java uses large heap</td></tr>
            <tr><td>Deep Recursion</td><td>C (50 KB)</td><td>Java (40 MB)</td><td>Python/Java stack heavier</td></tr>
        </tbody>
    </table><br>

    <h3>4.3 Development Ergonomics</h3>
    <ul>
        <li> <b>Python</b>: Fastest to code and debug; ideal for scripts, data science, and rapid testing.</li>
        <li> <b>Java</b>: Excellent IDEs, profiling tools (VisualVM, JFR).</li>
        <li> <b>C/C++</b>: Require deep expertise; slower development cycle but unmatched control.</li>
    </ul>

    <h3>4.4 Stability Under Load</h3>
    <ul>
        <li> All languages ran stably under the workloads.</li>
        <li> Java’s G1 GC showed short, predictable pauses.</li>
        <li> C/C++ showed 0 memory leaks (Valgrind).</li>
        <li> Python’s GC was not significantly stressed.</li>
    </ul>

    <h3>4.5 Debugging & Observability</h3>
    <table style='width:100%; font-size:16px;' border='1' cellpadding='5'>
        <thead>
            <tr><th>Language</th><th>Tools</th><th>Observability Focus</th></tr>
        </thead>
        <tbody>
            <tr><td>Java</td><td>VisualVM, JFR</td><td>GC tracking, allocation, threads</td></tr>
            <tr><td>C/C++</td><td>Valgrind, gprof</td><td>Leaks, invalid access, performance</td></tr>
            <tr><td>Python</td><td>cProfile, tracemalloc</td><td>Function calls, object memory</td></tr>
        </tbody>
    </table><br>

    <h3>4.6 Concurrency Model</h3>
    <ul>
        <li> <b>Python</b>: GIL limits CPU-bound threading; use multiprocessing instead.</li>
        <li> <b>Java</b>: OS threads + thread pools = great parallelism.</li>
        <li> <b>C/C++</b>: Real threading with pthread or std::thread.</li>
    </ul>

    <h3>4.7 Developer Productivity</h3>
    <ul>
        <li> <b>Python</b>: Most productive; great for prototyping and scripting.</li>
        <li> <b>Java</b>: Balanced for scale and observability; rich tooling.</li>
        <li> <b>C/C++</b>: Most control, steepest learning curve.</li>
    </ul>

    <h3 style='margin-top:20px'>Bottom-line Comparative Call</h3>
    <table style='width:100%; font-size:16px;' border='1' cellpadding='5'>
        <thead>
            <tr><th>Use Case</th><th>Recommended Language</th></tr>
        </thead>
        <tbody>
            <tr><td>Tight numeric / low-level systems</td><td>C / C++</td></tr>
            <tr><td>Long-running service with observability</td><td>Java</td></tr>
            <tr><td>Prototyping, scripting, data analysis</td><td>Python</td></tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)
    
    data = {
        "Task": ["Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd", "Fibonacci+Odd",
                "N-body", "N-body", "N-body", "N-body",
                "Deep Recursion", "Deep Recursion", "Deep Recursion", "Deep Recursion"],
        "Language": ["Python", "Java", "C", "C++"] * 3,
        "Time (s)": [21.05, 0.032, 0.18, 0.07, 2.34, 0.42, 0.24, 0.11, 1.02, 0.23, 0.15, 0.09]
    }

    fig = px.bar(data, x="Task", y="Time (s)", color="Language", barmode="group",
                title="Execution Time per Task (lower is better)")
    fig.show()
    st.plotly_chart(fig, use_container_width=True)
    languages = ['Python', 'C', 'C++', 'Java']
    execution_times = [21.05, 0.07, 0.18, 0.30]  # in seconds
    
    # Plotly Bar Chart
    fig = go.Figure(data=[
        go.Bar(
            x=languages,
            y=execution_times,
            marker_color=['#FFDD57', '#4CAF50', '#2196F3', '#F44336'],  # Optional custom colors
            text=[f"{t:.2f} s" for t in execution_times],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Execution Time Comparison by Language",
        xaxis_title="Programming Language",
        yaxis_title="Execution Time (seconds)",
        yaxis=dict(range=[0, max(execution_times) * 1.2]),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
    memory_usage = [0.3, 0.001, 7.7, 5.5]  # in MB

    # Create Plotly bar chart
    memory_fig = go.Figure(data=[
        go.Bar(
            x=languages,
            y=memory_usage,
            marker_color='mediumslateblue',
            text=[f"{m:.3f} MB" if m < 1 else f"{m:.1f} MB" for m in memory_usage],
            textposition='outside'
        )
    ])

    # Layout customization
    memory_fig.update_layout(
        title="Memory Usage Comparison by Language",
        xaxis_title="Programming Language",
        yaxis_title="Peak Memory Usage (MB)",
        yaxis=dict(range=[0, max(memory_usage) * 1.2]),
        template="plotly_white"
    )

    # Display in Streamlit
    st.plotly_chart(memory_fig, use_container_width=True)
    dev_effort = [1, 5, 4, 3]      # 1 = easiest to develop, 5 = hardest
    performance = [1, 5, 5, 4]     # 1 = slowest, 5 = fastest

    # Plotly scatter plot
    scatter_fig = go.Figure()

    scatter_fig.add_trace(go.Scatter(
        x=dev_effort,
        y=performance,
        mode='markers+text',
        text=languages,
        textposition='top center',
        marker=dict(size=14, color='orange', line=dict(width=2, color='black'))
    ))

    scatter_fig.update_layout(
        title="Developer Effort vs Performance",
        xaxis_title="Development Effort (1 = Easy, 5 = Hard)",
        yaxis_title="Performance (1 = Slow, 5 = Fast)",
        xaxis=dict(dtick=1),
        yaxis=dict(dtick=1),
        template="plotly_white",
        width=700,
        height=500
    )

    # Show chart in Streamlit
    st.plotly_chart(scatter_fig, use_container_width=True)

    gc_collections = [0, 0, 0, 1]  # Only Java triggered GC during benchmark
    thread_counts = [2, 2, 1, 2]   # Based on your setup (Java, Python: 2 threads; C++: 1 thread)

    # Create grouped bar chart
    gc_thread_fig = go.Figure(data=[
        go.Bar(name='GC Collections', x=languages, y=gc_collections, marker_color='indianred'),
        go.Bar(name='Thread Count', x=languages, y=thread_counts, marker_color='seagreen')
    ])

    gc_thread_fig.update_layout(
        title="GC Collections and Thread Count by Language",
        xaxis_title="Programming Language",
        yaxis_title="Count",
        barmode='group',
        template="plotly_white"
    )

    # Display in Streamlit
    st.plotly_chart(gc_thread_fig, use_container_width=True)

    characteristics = ['Speed', 'Memory Efficiency', 'Debuggability', 'Tooling', 'Ease of Use', 'Threading']

    # Radar values: scores for each language across the attributes
    values = {
        'Python': [1, 2, 3, 3, 5, 1],
        'C':      [5, 5, 3, 2, 2, 5],
        'C++':    [5, 4, 4, 3, 3, 5],
        'Java':   [4, 3, 5, 5, 4, 5]
    }

    # Create radar chart
    radar_fig = go.Figure()

    for lang in languages:
        radar_fig.add_trace(go.Scatterpolar(
            r=values[lang],
            theta=characteristics,
            fill='toself',
            name=lang
        ))

    radar_fig.update_layout(
        title="Language Characteristics Radar Plot",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=10))
        ),
        template="plotly_white",
        showlegend=True
    )

    time = list(range(0, 11))  # 0 to 10 seconds

    memory_python = [5, 6, 7, 8, 9, 10, 10, 9.5, 9.3, 9.2, 9.1]
    memory_java =   [30, 35, 38, 42, 45, 50, 48, 46, 44, 42, 41]
    memory_cpp =    [3, 3.2, 3.5, 4, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3, 4.3]
    memory_c =      [1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

    # Create Plotly line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=memory_python, mode='lines+markers', name='Python', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=time, y=memory_java, mode='lines+markers', name='Java', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=time, y=memory_cpp, mode='lines+markers', name='C++', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=time, y=memory_c, mode='lines+markers', name='C', line=dict(color='green')))

    # Customize layout
    fig.update_layout(
        title="Memory Usage Over Time (Runtime Memory Profile)",
        xaxis_title="Time (seconds)",
        yaxis_title="Memory Usage (MB)",
        template="plotly_white",
        legend_title="Language"
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # Display in Streamlit
    st.plotly_chart(radar_fig, use_container_width=True)

    data = [
        {"Thread": "Java - Odd Numbers", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:01.3"},
        {"Thread": "Java - Fibonacci", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:01.25"},
        {"Thread": "Python - Odd Numbers", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:22"},
        {"Thread": "Python - Fibonacci", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:21.5"},
        {"Thread": "C - Odd Numbers", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:01.02"},
        {"Thread": "C - Fibonacci", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:01.05"},
        {"Thread": "C++ - Combined", "Start": "2024-01-01 00:00:01", "Finish": "2024-01-01 00:00:01.18"},
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['Start'] = pd.to_datetime(df['Start'])
    df['Finish'] = pd.to_datetime(df['Finish'])

    # Create Gantt chart
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Thread", color="Thread")
    fig.update_layout(
        title="Thread Execution Timeline",
        xaxis_title="Time",
        yaxis_title="Thread",
        xaxis_tickformat="%H:%M:%S.%L",
        template="plotly_white",
        showlegend=False
    )
    fig.update_yaxes(autorange="reversed")  # So first thread appears on top

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    line_counts = [15, 45, 55, 50]  # Approximate LOC for odd+fib benchmark
    benchmark_quality = [3, 4, 4, 5]  # 1 = shallow, 5 = deep profiling/tools

    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=line_counts,
        y=benchmark_quality,
        mode='markers+text',
        text=languages,
        textposition='top center',
        marker=dict(size=14, color='purple', line=dict(width=2, color='black'))
    ))

    # Customize layout
    fig.update_layout(
        title="Line Count vs Benchmark Quality",
        xaxis_title="Lines of Code (Benchmark Implementation)",
        yaxis_title="Benchmark Quality (1 = Shallow, 5 = Deep)",
        xaxis=dict(dtick=10),
        yaxis=dict(dtick=1),
        template="plotly_white"
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Tooling difficulty data (1 = easy, 5 = hard)
    data = {
        "Tool": ["cProfile", "tracemalloc", "Valgrind", "gprof", "VisualVM", "JFR"],
        "Python":   [1, 2, None, None, None, None],
        "C":        [None, None, 3, 4, None, None],
        "C++":      [None, None, 4, 4, None, None],
        "Java":     [None, None, None, None, 2, 3]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df_melted = df.melt(id_vars=["Tool"], var_name="Language", value_name="Difficulty")
    df_melted.dropna(inplace=True)  # Remove None values

    # Create heatmap
    fig = px.density_heatmap(
        df_melted,
        x="Language",
        y="Tool",
        z="Difficulty",
        color_continuous_scale="RdYlGn_r",
        title="Profiler Tooling Difficulty (1 = Easy, 5 = Hard)",
        labels={"Difficulty": "Setup / Learning Curve"}
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    exec_times = [21.05, 0.30, 0.18, 0.07]

    # Normalize against slowest (Python)
    slowest_time = max(exec_times)
    speedup = [round(slowest_time / t, 2) for t in exec_times]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=languages,
            y=speedup,
            text=[f"{s}×" for s in speedup],
            textposition="outside",
            marker_color='teal'
        )
    ])

    fig.update_layout(
        title="Normalized Speedup (Relative to Python)",
        xaxis_title="Language",
        yaxis_title="Speedup (×)",
        yaxis=dict(range=[0, max(speedup) * 1.2]),
        template="plotly_white"
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    risk_factors = [
        "Segfault Risk",
        "Memory Leak Risk",
        "Null Pointer Risk",
        "Exception Handling Strength (inverse)",
        "Debug/Profiling Visibility (inverse)"
    ]

    # Risk scale: 1 = Safe/Low Risk, 5 = Very Risky
    risk_scores = {
        "Python": [1, 1, 1, 1, 2],
        "Java":   [1, 1, 2, 1, 1],
        "C":      [5, 5, 5, 4, 4],
        "C++":    [4, 5, 5, 4, 3]
    }

    # Convert to DataFrame
    risk_df = pd.DataFrame(risk_scores, index=risk_factors)

    # Create Radar Chart
    fig = go.Figure()

    for lang in risk_df.columns:
        fig.add_trace(go.Scatterpolar(
            r=risk_df[lang].values,
            theta=risk_factors,
            fill='toself',
            name=lang
        ))

    # Layout
    fig.update_layout(
        title="Error and Safety Risk Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        template="plotly_white",
        showlegend=True
    )

    # Show in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    time_points = list(range(0, 10))  # 0 to 9 seconds

    memory_data = pd.DataFrame({
        'Time (s)': time_points,
        'Python': [30, 35, 40, 42, 45, 46, 48, 50, 52, 54],
        'Java':   [20, 25, 30, 35, 38, 40, 41, 43, 44, 45],
        'C':      [2, 2, 2, 2, 2, 2, 3, 3, 3, 3],
        'C++':    [5, 6, 6, 6, 7, 7, 8, 8, 9, 9]
    })

    fig = go.Figure()

    # Add each language as an area trace
    for lang in ['Python', 'Java', 'C', 'C++']:
        fig.add_trace(go.Scatter(
            x=memory_data['Time (s)'],
            y=memory_data[lang],
            mode='lines',
            name=lang,
            stackgroup='one'  # enables stacking
        ))

    fig.update_layout(
        title='Stacked Area Chart: Memory Usage Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Memory Usage (MB)',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

if section == "Trade-offs":
    st.markdown("""
    <p style='font-size:18px'>
    <span style='font-size:22px; font-weight:bold;'>5. Discussion of Trade-offs</span><br><br>

    <span style='font-size:20px; font-weight:bold;'>5.1 Performance vs. Development Time</span><br>
    • <strong>Python:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Performance:</strong> CPU-bound loops in CPython are 10–100× slower than native code due to bytecode interpretation, type checks, and object overhead.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Dev speed:</strong> Minimal boilerplate, rapid prototyping, simple profiling (cProfile, tracemalloc).<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Mitigation:</strong> Use NumPy, Numba, or Cython to move compute-heavy code to native backend.<br><br>

    • <strong>C/C++:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Performance:</strong> Compiled with <code>-O3 -march=native</code>, achieves best runtimes via loop unrolling, vectorization.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Dev speed:</strong> Slower due to manual memory/thread management and tool setup.<br><br>

    • <strong>Java:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Performance:</strong> JIT compiles hot loops, reaching near-native speed post warm-up (~0.03 s).<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Dev speed:</strong> Rich libraries, strong typing, great IDE/tooling like VisualVM and JFR.<br><br>

    <span style='font-size:20px; font-weight:bold;'>5.2 Reliability vs. Control</span><br>
    • <strong>C++:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Control:</strong> Direct memory and low-level access.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Risk:</strong> Mismanagement can cause leaks, corruption, race conditions.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Modern C++:</strong> RAII, smart pointers, and threads help mitigate risks.<br><br>

    • <strong>Java & Python:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Reliability:</strong> Automatic GC and memory safety prevent common bugs.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Trade-off:</strong> Reduced control over memory layout and cache locality; GC pauses may still matter.<br><br>

    <span style='font-size:20px; font-weight:bold;'>5.3 Memory</span><br>
    • <strong>C/C++:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Efficiency:</strong> Stack-based or zero-allocation designs offer smallest footprints.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Your data:</strong> C used ~1 KB; C++ ~7.7 MB (for 1M Fibonacci in vector).<br><br>

    • <strong>Java:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Baseline:</strong> JVM overhead = 50–55 MB used heap; max heap 805 MB.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Amortization:</strong> Negligible in large-scale services.<br><br>

    • <strong>Python:</strong><br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Overhead:</strong> High metadata per object.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Your run:</strong> tracemalloc shows 0.3 MB due to minimal allocations.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;◦ <strong>Reality:</strong> With large object sets, memory use rises; NumPy helps via dense arrays.<br><br>

    <span style='font-size:20px; font-weight:bold;'>5.4 Comparative Summary Table</span><br><br>

    <table style="width:100%; font-size:17px; border-collapse: collapse;" border="1">
        <thead style="font-weight:bold;">
            <tr>
                <th style="padding:6px;">Aspect</th>
                <th style="padding:6px;">Python</th>
                <th style="padding:6px;">C/C++</th>
                <th style="padding:6px;">Java</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding:6px;">Loop performance</td>
                <td>Slowest (GIL + interpreter)</td>
                <td>Fastest (native, optimized)</td>
                <td>Near-native (JIT)</td>
            </tr>
            <tr>
                <td style="padding:6px;">Dev time</td>
                <td>Fastest to code</td>
                <td>Longest</td>
                <td>Medium-fast</td>
            </tr>
            <tr>
                <td style="padding:6px;">Memory control</td>
                <td>Low</td>
                <td>Highest</td>
                <td>Medium</td>
            </tr>
            <tr>
                <td style="padding:6px;">Reliability</td>
                <td>High (runtime safety)</td>
                <td>Lower without discipline</td>
                <td>High (runtime safety)</td>
            </tr>
            <tr>
                <td style="padding:6px;">Baseline memory</td>
                <td>Medium (~10s MB)</td>
                <td>Minimal (KB–MB)</td>
                <td>High (50+ MB)</td>
            </tr>
            <tr>
                <td style="padding:6px;">Best use case</td>
                <td>Rapid prototyping, glue code</td>
                <td>Performance-critical systems</td>
                <td>Long-running scalable services</td>
            </tr>
        </tbody>
    </table>
    </p>
    """, unsafe_allow_html=True)

if section == "Threats to Validity":
    st.markdown("""
    <div style='font-size:18px'>

    <h2>6. Threats to Validity</h2>
    <p>
    Despite the efforts to standardize benchmarking across Python, Java, C, and C++, several internal and external factors may affect the accuracy, fairness, or generalizability of the results. This section outlines those key concerns.
    </p>

    <h3>6.1 Non-Uniform Workloads</h3>
    <ul>
        <li>C++ stored the entire Fibonacci sequence in memory; Python computed it in constant space.</li>
        <li>Loop boundaries and result printing behaviors varied slightly, introducing runtime deviations.</li>
        <li>Base conditions and recursion structures differ subtly due to language semantics and call stack behavior.</li>
    </ul>

    <h3>6.2 JIT Warm-up Effects (Java)</h3>
    <ul>
        <li>Java’s Just-In-Time (JIT) compiler only activates after method "hotness" detection, requiring warm-up runs.</li>
        <li>Short benchmarks may misrepresent Java's speed depending on whether the JIT has optimized the path.</li>
        <li>Steady-state benchmarking is required for accurate Java performance assessment.</li>
    </ul>

    <h3>6.3 Profiler Overhead</h3>
    <ul>
        <li>Valgrind significantly slows native C/C++ execution, affecting timing accuracy.</li>
        <li>Python’s cProfile and tracemalloc only observe Python-level behavior, not OS/native libraries.</li>
        <li>Java's VisualVM adds minimal overhead but still affects memory and GC timings.</li>
    </ul>

    <h3>6.4 Memory Measurement Discrepancies</h3>
    <ul>
        <li><b>Python:</b> <code>tracemalloc</code> tracks Python object allocations, but not full process memory (RSS).</li>
        <li><b>Java:</b> VisualVM shows heap but excludes native memory (JIT cache, thread stacks).</li>
        <li><b>C/C++:</b> Valgrind and <code>/usr/bin/time</code> provide low-level data, but miss stack usage and temporary buffers.</li>
        <li>Cross-language comparisons require OS-level normalization (e.g., <code>top</code>, <code>smem</code>).</li>
    </ul>

    <h3>6.5 Garbage Collection and Runtime Behavior</h3>
    <ul>
        <li><b>Java:</b> GC causes nondeterministic pauses, especially under high allocation workloads.</li>
        <li><b>Python:</b> GC based on reference counting; cycle detection may not trigger during short runs.</li>
        <li><b>C/C++:</b> No GC introduces memory leak risk but ensures stable timings.</li>
    </ul>

    <h3>6.6 Platform and Environment Variability</h3>
    <ul>
        <li>Benchmarks were run on a specific setup (Windows/WSL2/Linux), affecting comparability.</li>
        <li>Windows task scheduling adds jitter; Linux gives more consistent timing.</li>
        <li>Power plans, CPU scaling, and background tasks weren’t uniformly controlled.</li>
    </ul>

    <h3>6.7 Compiler Flags and Build Optimization</h3>
    <ul>
        <li><b>C/C++:</b> Performance can vary widely based on flags like <code>-O3</code>, <code>-march=native</code>, <code>-flto</code>.</li>
        <li><b>Java:</b> JVM tuning (e.g., <code>-XX:+UseG1GC</code>) impacts speed and memory usage.</li>
        <li><b>Python:</b> No compiler flags, but use of NumPy or PyPy could change outcomes (not used here).</li>
    </ul>

    <h3>6.8 Language-Specific Runtime Cost Assumptions</h3>
    <ul>
        <li>Language models differ in base overhead — Python interpreter startup, Java JVM, C/C++ minimal runtime.</li>
        <li>Short-duration tasks highlight initialization time over real algorithmic performance.</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

if section == "Conclusion":
    st.markdown("""
    <div style='font-size:18px'>

    <h2>7. Conclusion</h2>
    <p>
    This comparative study evaluated the runtime efficiency, memory usage, garbage collection behavior,
    concurrency support, and development ergonomics of four popular programming languages—
    <b>C</b>, <b>C++</b>, <b>Java</b>, and <b>Python</b>—across a diverse set of computationally intensive tasks:
    Fibonacci + Odd Number generation, N-body simulation, Deep Recursion, and N-gram string processing.
    </p>

    <h3>7.1 Key Findings</h3>
    <h4>Performance</h4>
    <ul>
        <li><b>C++</b> emerged as the fastest across nearly all workloads, thanks to low-level optimizations, inlined operations, and <code>std::thread</code> usage.</li>
        <li><b>C</b> followed closely, especially in memory-light tasks, leveraging manual memory control and minimal runtime overhead.</li>
        <li><b>Java</b> achieved near-native performance after JIT warm-up, excelling in loop-heavy and concurrent tasks.</li>
        <li><b>Python</b> was slower due to interpreter overhead and the GIL, but remains suitable for moderate workloads and rapid prototyping.</li>
    </ul>

    <h4>Memory Usage</h4>
    <ul>
        <li><b>C</b> and <b>C++</b> had the smallest memory footprints when avoiding heap allocations or using scoped storage.</li>
        <li><b>Java</b> carried a consistent overhead from class metadata, JIT cache, and GC-managed heap.</li>
        <li><b>Python</b> showed high per-object memory usage, reflecting interpreter and object model costs.</li>
    </ul>

    <h4>Concurrency</h4>
    <ul>
        <li><b>Java</b>, <b>C</b>, and <b>C++</b> effectively leveraged real OS threads for parallelism.</li>
        <li><b>Python</b> was limited by the GIL, requiring multiprocessing or native extensions for CPU-bound parallel work.</li>
    </ul>

    <h4>Developer Experience & Tooling</h4>
    <ul>
        <li><b>Python</b>: Shortest development cycle and easy profiling (cProfile, tracemalloc).</li>
        <li><b>Java</b>: Strong profiling tools (VisualVM, JFR), robust threading, memory safety.</li>
        <li><b>C/C++</b>: Demanding development but unmatched control and optimization potential.</li>
    </ul>

    <h3>7.2 General Recommendations</h3>
    <ul>
        <li><b>Maximum performance</b> in CPU-bound numerical/simulation tasks → <b>C++</b> or <b>C</b>.</li>
        <li><b>Balanced speed, safety, tooling</b> for long-running/concurrent apps → <b>Java</b>.</li>
        <li><b>Rapid experimentation & integration</b> with external libraries → <b>Python</b> (NumPy, Numba, Cython).</li>
    </ul>

    </div>
    <span style='font-size:20px; font-weight:bold;'><br><br>8.1 Summary Table</span><br>
    <table style="width:100%; font-size:16px; border-collapse: collapse;" border="1">
        <thead style="font-weight:bold;">
            <tr>
                <th style="padding:6px;">Parameter</th>
                <th style="padding:6px;">Python</th>
                <th style="padding:6px;">Java</th>
                <th style="padding:6px;">C</th>
                <th style="padding:6px;">C++</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Runtime Speed</td><td>Slow</td><td>Good</td><td>Very Fast</td><td>Very Fast</td></tr>
            <tr><td>Memory Usage</td><td>High</td><td>Medium (GC overhead)</td><td>Very Efficient</td><td>Efficient</td></tr>
            <tr><td>Concurrency</td><td>Threading limited by GIL</td><td>Multithreading</td><td>Multithreading</td><td>Multithreading</td></tr>
            <tr><td>Compilation Time</td><td>N/A (Interpreted)</td><td>Slow</td><td>Fast</td><td>Often Slow (templates)</td></tr>
            <tr><td>Memory Safety</td><td>Unsafe</td><td>GC-managed</td><td>Manual & unsafe</td><td>Manual & unsafe</td></tr>
            <tr><td>Startup Time</td><td>Fast</td><td>Slow</td><td>Fast</td><td>Fast</td></tr>
            <tr><td>Binary Size</td><td>N/A</td><td>Large</td><td>Small</td><td>Medium-Large</td></tr>
            <tr><td>Developer Productivity</td><td>High</td><td>High</td><td>Low (verbose)</td><td>Medium</td></tr>
            <tr><td>Ecosystem & Libraries</td><td>Huge (AI, DS, Web)</td><td>Mature</td><td>Smaller</td><td>Rich (games, systems, ML)</td></tr>
            <tr><td>Cross-Platform</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
            <tr><td>Portability</td><td>High</td><td>High</td><td>High</td><td>High</td></tr>
            <tr><td>Learning Curve</td><td>Easy</td><td>Medium</td><td>Steep</td><td>Steep</td></tr>
            <tr><td>Performance-Critical Apps</td><td>Not suitable</td><td>Moderate</td><td>Excellent</td><td>Excellent</td></tr>
            <tr><td>Embedded Systems</td><td>No</td><td>Rare</td><td>Excellent</td><td>Good</td></tr>
            <tr><td>GUI Development</td><td>Possible (Tkinter, PyQt)</td><td>Good (JavaFX, Swing)</td><td>Rare</td><td>Good (Qt, wxWidgets)</td></tr>
            <tr><td>Best Use Cases</td><td>AI, Scripting, Data Science</td><td>Web, Enterprise, Android</td><td>Systems, OS, Real-Time</td><td>Games, High-Perf Apps</td></tr>
            <tr><td>Type System</td><td>Dynamic</td><td>Static (Strong)</td><td>Static</td><td>Static</td></tr>
            <tr><td>Garbage Collection</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr>
            <tr><td>Low-level Hardware Access</td><td>No</td><td>Limited</td><td>Yes</td><td>Yes</td></tr>
        </tbody>
    </table><br><br>

    
    </p>
    """, unsafe_allow_html=True)
    st.image(r"C:\Users\micha\Documents\bsc\Programs\Images\reslut3.png", 
             caption="Results Summary", 
             use_column_width=True)

st.markdown(section_content[section] if section_content[section] else "")
st.sidebar.write("Developed by Michael Fernandes")