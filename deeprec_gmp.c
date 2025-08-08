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
