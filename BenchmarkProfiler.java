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
