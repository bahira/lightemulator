#include <stdio.h>
#include <math.h>

// Standard LogSumExp for comparison
float standard_logsumexp(const float *x, size_t n) {
    float max = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] > max) max = x[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += expf(max - x[i]);  // Note: this is exp(x_i - max) but we want expf(max - x_i)...
    }
    // Actually correct: sum += expf(x[i] - max);
    return max + logf(sum / n);  // This is wrong too, let me do it properly
}

// Correct standard LogSumExp
float standard_logsumexp_correct(const float *x, size_t n) {
    float max = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] > max) max = x[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += expf(x[i] - max);
    }
    return max + logf(sum);
}

int main(void) {
    float data[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        2.0f, 1.0f, 0.0f, -1.0f, 3.0f, 2.0f, 1.0f, 0.0f
    };
    
    printf("Standard LSE: %.10f\n", standard_logsumexp_correct(data, 16));
    
    return 0;
}