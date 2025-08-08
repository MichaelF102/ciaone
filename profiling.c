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
    printf("\n✅ Odd numbers (1M) computed.\n");
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
    printf("\n✅ Fibonacci (1M) computed. Last: %llu\n", fib[n - 1]);
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
    printf("\n⏱️ Total Execution Time: %.6f seconds\n", elapsed);

    return 0;
}
