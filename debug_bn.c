/* Debug BT30 big number multiply */
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

void spear_bignum_mul_schoolbook(
    const uint32_t *a, const uint32_t *b,
    uint32_t *out, size_t n_words)
{
    for (size_t k = 0; k < 2 * n_words; ++k) {
        out[k] = 0;
    }

    for (size_t i = 0; i < n_words; ++i) {
        for (size_t j = 0; j < n_words; ++j) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            size_t pos = i + j;
            uint64_t sum = (uint64_t)out[pos] + prod;
            out[pos] = (uint32_t)sum;
            out[pos + 1] += (uint32_t)(sum >> 32);
        }
    }

    for (size_t k = 0; k < 2 * n_words - 1; ++k) {
        if (out[k] > 0xFFFFFFFF) {
            out[k + 1] += 1;
            out[k] &= 0xFFFFFFFF;
        }
    }
}

void spear_bignum_print(const uint32_t *num, size_t n_words)
{
    int found_nonzero = 0;
    for (size_t i = 2 * n_words; i > 0; --i) {
        uint32_t w = num[i-1];
        if (w != 0) {
            if (!found_nonzero) {
                printf("%08" PRIx32, w);
                found_nonzero = 1;
            } else {
                printf("%08" PRIx32, w);
            }
        } else if (found_nonzero) {
            printf("%08" PRIx32, w);
        }
    }
    if (!found_nonzero) {
        printf("0");
    }
    printf("\n");
}

int main(void) {
    /* Test 2: 255 * 255 */
    uint32_t a[2] = {0xff, 0};
    uint32_t b[2] = {0xff, 0};
    uint32_t out[4] = {0};

    spear_bignum_mul_schoolbook(a, b, out, 2);

    printf("out[0] = %" PRIu32 " = 0x%08" PRIx32 "\n", out[0], out[0]);
    printf("out[1] = %" PRIu32 " = 0x%08" PRIx32 "\n", out[1], out[1]);
    printf("out[2] = %" PRIu32 " = 0x%08" PRIx32 "\n", out[2], out[2]);
    printf("out[3] = %" PRIu32 " = 0x%08" PRIx32 "\n", out[3], out[3]);

    printf("Print function output: ");
    spear_bignum_print(out, 2);

    return 0;
}