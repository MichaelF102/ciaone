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
            System.out.println("❌ StackOverflowError at depth: " + maxDepth);
        } catch (Throwable t) {
            System.out.println("❌ Exception: " + t);
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
