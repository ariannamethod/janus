/*
 * test_sigmoid_gate.c — Verify sigmoid attention gate behavior
 *
 * Tests:
 *   1. Gate outputs are each independently in [0,1]
 *   2. Gate outputs do NOT sum to 1 (unlike softmax)
 *   3. Changing one gate input does NOT affect other gate outputs
 *   4. Edge cases: large positive, large negative, zero
 *
 * Compile: cc -O2 -std=c11 -o test_sigmoid_gate tests/test_sigmoid_gate.c -lm
 * Run:     ./test_sigmoid_gate
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* === Functions under test (copied from janus.c) === */

static float sigmoidf(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static void gate_sigmoid(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = sigmoidf(x[i]);
}

/* Softmax for comparison */
static void row_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    if (s > 0) for (int i = 0; i < n; i++) x[i] /= s;
}

/* === Test infrastructure === */

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
    } else { \
        tests_passed++; \
    } \
} while(0)

#define EPS 1e-6f

/* === Test 1: Gate outputs are each in [0,1] independently === */
static void test_sigmoid_gate_range(void) {
    printf("test_sigmoid_gate_range:\n");

    float inputs[][3] = {
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  3.0f},
        {-1.0f, -2.0f, -3.0f},
        { 5.0f, -5.0f,  0.0f},
        {10.0f, 10.0f, 10.0f},
        {-10.0f,-10.0f,-10.0f},
        {50.0f, -50.0f, 0.0f},
    };
    int n_tests = (int)(sizeof(inputs) / sizeof(inputs[0]));

    for (int t = 0; t < n_tests; t++) {
        float g[3];
        memcpy(g, inputs[t], sizeof(g));
        gate_sigmoid(g, 3);

        for (int i = 0; i < 3; i++) {
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "input[%d]=(%.1f,%.1f,%.1f) gate[%d]=%.6f in [0,1]",
                     t, inputs[t][0], inputs[t][1], inputs[t][2], i, g[i]);
            ASSERT(g[i] >= 0.0f && g[i] <= 1.0f, msg);
        }
    }
    printf("  passed range checks\n");
}

/* === Test 2: Gate outputs do NOT sum to 1 (unlike softmax) === */
static void test_sigmoid_not_normalized(void) {
    printf("test_sigmoid_not_normalized:\n");

    float inputs[][3] = {
        { 1.0f,  2.0f,  3.0f},
        { 5.0f,  5.0f,  5.0f},
        {-1.0f,  0.0f,  1.0f},
    };
    int n_tests = (int)(sizeof(inputs) / sizeof(inputs[0]));

    int non_one_count = 0;
    for (int t = 0; t < n_tests; t++) {
        float g[3];
        memcpy(g, inputs[t], sizeof(g));
        gate_sigmoid(g, 3);
        float sum = g[0] + g[1] + g[2];

        if (fabsf(sum - 1.0f) > 0.01f) non_one_count++;

        char msg[128];
        snprintf(msg, sizeof(msg),
                 "input=(%.1f,%.1f,%.1f) sigmoid sum=%.4f",
                 inputs[t][0], inputs[t][1], inputs[t][2], sum);
        printf("  %s\n", msg);

        /* Verify softmax WOULD sum to 1 */
        float s[3];
        memcpy(s, inputs[t], sizeof(s));
        row_softmax(s, 3);
        float ssum = s[0] + s[1] + s[2];
        snprintf(msg, sizeof(msg),
                 "softmax sum=%.6f equals 1.0", ssum);
        ASSERT(fabsf(ssum - 1.0f) < EPS, msg);
    }

    ASSERT(non_one_count >= 2,
           "sigmoid sums should generally NOT equal 1.0");

    /* All gates high: sum > 1 */
    float high[3] = {5.0f, 5.0f, 5.0f};
    gate_sigmoid(high, 3);
    float hsum = high[0] + high[1] + high[2];
    char msg[128];
    snprintf(msg, sizeof(msg),
             "all-high gates sum=%.4f > 1.0 (all mechanisms active)", hsum);
    ASSERT(hsum > 2.5f, msg);
    printf("  %s\n", msg);

    /* All gates low: sum < 1 */
    float low[3] = {-5.0f, -5.0f, -5.0f};
    gate_sigmoid(low, 3);
    float lsum = low[0] + low[1] + low[2];
    snprintf(msg, sizeof(msg),
             "all-low gates sum=%.4f < 1.0 (all mechanisms suppressed)", lsum);
    ASSERT(lsum < 0.1f, msg);
    printf("  %s\n", msg);
}

/* === Test 3: Changing one gate input does NOT affect other outputs === */
static void test_gate_independence(void) {
    printf("test_gate_independence:\n");

    float base[3] = {1.0f, 2.0f, 3.0f};
    float modified[3] = {1.0f, 2.0f, 3.0f};

    /* Compute base sigmoid */
    float g_base[3];
    memcpy(g_base, base, sizeof(g_base));
    gate_sigmoid(g_base, 3);

    /* Change gate[0] input drastically */
    modified[0] = -10.0f;
    float g_mod[3];
    memcpy(g_mod, modified, sizeof(g_mod));
    gate_sigmoid(g_mod, 3);

    /* Gate[1] and gate[2] should be UNCHANGED */
    char msg[128];
    snprintf(msg, sizeof(msg),
             "sigmoid: changing gate[0] -> gate[1] unchanged (%.6f == %.6f)",
             g_base[1], g_mod[1]);
    ASSERT(fabsf(g_base[1] - g_mod[1]) < EPS, msg);
    printf("  %s\n", msg);

    snprintf(msg, sizeof(msg),
             "sigmoid: changing gate[0] -> gate[2] unchanged (%.6f == %.6f)",
             g_base[2], g_mod[2]);
    ASSERT(fabsf(g_base[2] - g_mod[2]) < EPS, msg);
    printf("  %s\n", msg);

    /* Verify softmax does NOT have this property */
    float s_base[3], s_mod[3];
    memcpy(s_base, base, sizeof(s_base));
    memcpy(s_mod, modified, sizeof(s_mod));
    row_softmax(s_base, 3);
    row_softmax(s_mod, 3);

    snprintf(msg, sizeof(msg),
             "softmax: changing gate[0] DOES affect gate[1] (%.6f != %.6f)",
             s_base[1], s_mod[1]);
    ASSERT(fabsf(s_base[1] - s_mod[1]) > 0.01f, msg);
    printf("  %s\n", msg);

    snprintf(msg, sizeof(msg),
             "softmax: changing gate[0] DOES affect gate[2] (%.6f != %.6f)",
             s_base[2], s_mod[2]);
    ASSERT(fabsf(s_base[2] - s_mod[2]) > 0.01f, msg);
    printf("  %s\n", msg);

    /* Test changing gate[1] */
    float mod2[3] = {1.0f, -5.0f, 3.0f};
    float g_mod2[3];
    memcpy(g_mod2, mod2, sizeof(g_mod2));
    gate_sigmoid(g_mod2, 3);

    snprintf(msg, sizeof(msg),
             "sigmoid: changing gate[1] -> gate[0] unchanged (%.6f == %.6f)",
             g_base[0], g_mod2[0]);
    ASSERT(fabsf(g_base[0] - g_mod2[0]) < EPS, msg);
    printf("  %s\n", msg);

    snprintf(msg, sizeof(msg),
             "sigmoid: changing gate[1] -> gate[2] unchanged (%.6f == %.6f)",
             g_base[2], g_mod2[2]);
    ASSERT(fabsf(g_base[2] - g_mod2[2]) < EPS, msg);
    printf("  %s\n", msg);
}

/* === Test 4: Edge cases === */
static void test_sigmoid_edge_cases(void) {
    printf("test_sigmoid_edge_cases:\n");

    /* Zero input -> 0.5 */
    float z = sigmoidf(0.0f);
    char msg[128];
    snprintf(msg, sizeof(msg), "sigmoid(0) = %.6f == 0.5", z);
    ASSERT(fabsf(z - 0.5f) < EPS, msg);
    printf("  %s\n", msg);

    /* Large positive -> ~1.0 */
    float lp = sigmoidf(100.0f);
    snprintf(msg, sizeof(msg), "sigmoid(100) = %.6f == 1.0", lp);
    ASSERT(fabsf(lp - 1.0f) < EPS, msg);

    /* Large negative -> ~0.0 */
    float ln = sigmoidf(-100.0f);
    snprintf(msg, sizeof(msg), "sigmoid(-100) = %.6f == 0.0", ln);
    ASSERT(fabsf(ln) < EPS, msg);

    /* Symmetry: sigmoid(x) + sigmoid(-x) = 1.0 */
    for (float x = -5.0f; x <= 5.0f; x += 0.5f) {
        float sx = sigmoidf(x);
        float snx = sigmoidf(-x);
        snprintf(msg, sizeof(msg),
                 "sigmoid(%.1f) + sigmoid(%.1f) = %.6f == 1.0", x, -x, sx + snx);
        ASSERT(fabsf(sx + snx - 1.0f) < EPS, msg);
    }
    printf("  passed edge cases + symmetry\n");
}

/* === Test 5: All mechanisms can be simultaneously active === */
static void test_all_active(void) {
    printf("test_all_active:\n");

    /* Sigmoid allows ALL gates near 1.0 */
    float all_high[3] = {10.0f, 10.0f, 10.0f};
    gate_sigmoid(all_high, 3);

    char msg[128];
    for (int i = 0; i < 3; i++) {
        snprintf(msg, sizeof(msg),
                 "all-high: gate[%d] = %.6f > 0.99", i, all_high[i]);
        ASSERT(all_high[i] > 0.99f, msg);
    }
    printf("  all 3 mechanisms simultaneously active: Content=%.4f RRPRAM=%.4f Echo=%.4f\n",
           all_high[0], all_high[1], all_high[2]);

    /* Sigmoid allows ALL gates near 0.0 */
    float all_low[3] = {-10.0f, -10.0f, -10.0f};
    gate_sigmoid(all_low, 3);

    for (int i = 0; i < 3; i++) {
        snprintf(msg, sizeof(msg),
                 "all-low: gate[%d] = %.6f < 0.01", i, all_low[i]);
        ASSERT(all_low[i] < 0.01f, msg);
    }
    printf("  all 3 mechanisms simultaneously suppressed: Content=%.4f RRPRAM=%.4f Echo=%.4f\n",
           all_low[0], all_low[1], all_low[2]);

    /* Softmax always sums to 1 — mechanisms compete */
    float s_high[3] = {10.0f, 10.0f, 10.0f};
    row_softmax(s_high, 3);
    float s_sum = s_high[0] + s_high[1] + s_high[2];
    snprintf(msg, sizeof(msg),
             "softmax sum is always 1.0 (got %.6f)", s_sum);
    ASSERT(fabsf(s_sum - 1.0f) < EPS, msg);
    printf("  softmax forces competition: each gate = %.4f (sum=%.4f)\n",
           s_high[0], s_sum);
}

/* === Test 6: Initialization check === */
static void test_initialization(void) {
    printf("test_initialization:\n");

    /* janus.c initializes gate logits to 0.0 */
    float init[3] = {0.0f, 0.0f, 0.0f};
    gate_sigmoid(init, 3);

    char msg[128];
    for (int i = 0; i < 3; i++) {
        snprintf(msg, sizeof(msg),
                 "gate[%d] at init (logit=0) = %.6f == 0.5", i, init[i]);
        ASSERT(fabsf(init[i] - 0.5f) < EPS, msg);
    }
    printf("  at initialization: all gates = 0.5 (balanced)\n");

    /* Softmax gives 1/3 at zero init */
    float sinit[3] = {0.0f, 0.0f, 0.0f};
    row_softmax(sinit, 3);
    snprintf(msg, sizeof(msg),
             "softmax at init gives 1/3 = %.6f", sinit[0]);
    ASSERT(fabsf(sinit[0] - 1.0f/3.0f) < EPS, msg);
    printf("  softmax at init: 1/3 = %.6f (different from sigmoid 0.5)\n", sinit[0]);
}

int main(void) {
    printf("=== Janus Sigmoid Gate Tests ===\n\n");

    test_sigmoid_gate_range();
    test_sigmoid_not_normalized();
    test_gate_independence();
    test_sigmoid_edge_cases();
    test_all_active();
    test_initialization();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
