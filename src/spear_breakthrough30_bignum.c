/* ========================================================================== */
/* SPEAR BT30 : Micro-Kernel Big Number Multiply Schoolbook O(n²)           */
/* ========================================================================== */
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

/** Schoolbook Big Number multiply: a * b = out (out size = 2*n_words) */
void spear_bignum_mul_schoolbook(
    const uint32_t *a, const uint32_t *b,
    uint32_t *out, size_t n_words)
{
    /* Zero out the result (2*n_words words) */
    for (size_t k = 0; k < 2 * n_words; ++k) {
        out[k] = 0;
    }

    /* Schoolbook multiplication: for each a[i]*b[j], add to out[i+j] */
    for (size_t i = 0; i < n_words; ++i) {
        for (size_t j = 0; j < n_words; ++j) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            size_t pos = i + j;

            /* Add prod to out[pos] using 64-bit arithmetic */
            uint64_t sum = (uint64_t)out[pos] + prod;
            out[pos] = (uint32_t)sum;
            /* Carry into the next word: high 32 bits of the sum */
            out[pos + 1] += (uint32_t)(sum >> 32);
        }
    }

    /* Ripple carry propagation: propagate any remaining carries */
    for (size_t k = 0; k < 2 * n_words - 1; ++k) {
        if (out[k] > 0xFFFFFFFF) {
            out[k + 1] += 1;
            out[k] &= 0xFFFFFFFF;
        }
    }
}

/** Affiche un tableau bignum en format little-endian hexadécimal */
void spear_bignum_print(const uint32_t *num, size_t n_words)
{
    /* Affiche du mot le plus significatif au moins significatif */
    /* Pour un produit de n_words mots, le resultat fait 2*n_words mots */
    int found_nonzero = 0;
    
    /* Parcourir du mot le plus haut vers le plus bas */
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

/* Test simple */
int main(void) {
    /* Test 1: 3 * 5 = 15 avec 2 mots chacun */
    uint32_t a[2] = {3, 0};     /* = 3 */
    uint32_t b[2] = {5, 0};     /* = 5 */
    uint32_t out[4] = {0};      /* produit peut faire jusqu'à 4 mots */

    spear_bignum_mul_schoolbook(a, b, out, 2);

    printf("Test 1: 3 * 5 = 15 en bignum (2 mots x 2 mots): ");
    spear_bignum_print(out, 2); /* Attendu: 0000000f (15 en hex) */
    printf(" (expected: 0000000f)\n\n");

    /* Test 2: 255 * 255 = 65025 */
    uint32_t a2[2] = {0xff, 0};  /* = 255 */
    uint32_t b2[2] = {0xff, 0};  /* = 255 */
    uint32_t out2[4] = {0};

    spear_bignum_mul_schoolbook(a2, b2, out2, 2);

    printf("Test 2: 255 * 255 = 65025 en bignum (2 mots x 2 mots): ");
    spear_bignum_print(out2, 2); /* Attendu: 00010001 (65025 = 0x10001) */
    printf(" (expected: 00010001 = 65025)\n\n");

    /* Test 3: 123 * 456 = 56088 */
    uint32_t a3[2] = {123, 0};  
    uint32_t b3[2] = {456, 0};  
    uint32_t out3[4] = {0};

    spear_bignum_mul_schoolbook(a3, b3, out3, 2);

    printf("Test 3: 123 * 456 = 56088 en bignum: ");
    spear_bignum_print(out3, 2); /* Attendu: 00015e70 (56088 = 0x15E70) */
    printf(" (expected: 00015e70 = 56088)\n\n");

    /* Test 4: Plus grand - 0xFFFF * 0xFFFF = 0xFFFE0001 */
    uint32_t a4[2] = {0xFFFF, 0};  
    uint32_t b4[2] = {0xFFFF, 0};  
    uint32_t out4[4] = {0};

    spear_bignum_mul_schoolbook(a4, b4, out4, 2);

    printf("Test 4: 65535 * 65535 = 4294836225 en bignum: ");
    spear_bignum_print(out4, 2); /* Attendu: 0xFFFE0001 */
    printf(" (expected: 0xFFFE0001)\n\n");

    /* Test 5: 3 mots x 3 mots - 7*17*11*19*13*23 */
    uint32_t a5[3] = {7, 11, 13};  
    uint32_t b5[3] = {17, 19, 23};  
    uint32_t out5[6] = {0};

    spear_bignum_mul_schoolbook(a5, b5, out5, 3);

    printf("Test 5: (7*11*13) * (17*19*23) bignum 3x3: ");
    spear_bignum_print(out5, 3); /* produit = 5071061, hex: 0x61A205 */
    printf(" (produit = 5071061, hex: 0x61A205)\n");

    return 0;
}