#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define INPUT 784
#define H1 10
#define H2 10
#define OUTPUT 10

#define TRAIN_SAMPLES 60000
#define TEST_SAMPLES 10000

#define EPOCHS 10
#define LR 0.1f

static float X_train[TRAIN_SAMPLES][INPUT];
static int   Y_train[TRAIN_SAMPLES];

static float X_test[TEST_SAMPLES][INPUT];
static int   Y_test[TEST_SAMPLES];

static float W1[INPUT][H1], B1[H1];
static float W2[H1][H2],    B2[H2];
static float W3[H2][OUTPUT],B3[OUTPUT];

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float dsigmoid(float y) {
    return y * (1.0f - y);
}

float randf() {
    return ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
}

int load_mnist(const char *file, float X[][INPUT], int Y[], int max) {
    FILE *f = fopen(file, "r");
    if (!f) {
        printf("Failed to open %s\n", file);
        exit(1);
    }

    char line[16000];
    fgets(line, sizeof(line), f); 

    int n = 0;
    while (fgets(line, sizeof(line), f) && n < max) {
        char *tok = strtok(line, ",");
        Y[n] = atoi(tok);

        for (int i = 0; i < INPUT; i++) {
            tok = strtok(NULL, ",");
            X[n][i] = atof(tok) / 255.0f;
        }
        n++;
    }

    fclose(f);
    return n;
}

void forward(
    float x[INPUT],
    float a1[H1],
    float a2[H2],
    float a3[OUTPUT]
) {
    for (int j = 0; j < H1; j++) {
        float s = B1[j];
        for (int i = 0; i < INPUT; i++)
            s += x[i] * W1[i][j];
        a1[j] = sigmoid(s);
    }

    for (int j = 0; j < H2; j++) {
        float s = B2[j];
        for (int i = 0; i < H1; i++)
            s += a1[i] * W2[i][j];
        a2[j] = sigmoid(s);
    }

    for (int j = 0; j < OUTPUT; j++) {
        float s = B3[j];
        for (int i = 0; i < H2; i++)
            s += a2[i] * W3[i][j];
        a3[j] = sigmoid(s);
    }
}

void backward(
    float x[INPUT],
    int label,
    float a1[H1],
    float a2[H2],
    float a3[OUTPUT]
) {
    float d3[OUTPUT], d2[H2], d1[H1];

    for (int i = 0; i < OUTPUT; i++) {
        float t = (i == label);
        d3[i] = (a3[i] - t) * dsigmoid(a3[i]);
    }

    for (int i = 0; i < H2; i++) {
        float s = 0;
        for (int j = 0; j < OUTPUT; j++)
            s += W3[i][j] * d3[j];
        d2[i] = s * dsigmoid(a2[i]);
    }

    for (int i = 0; i < H1; i++) {
        float s = 0;
        for (int j = 0; j < H2; j++)
            s += W2[i][j] * d2[j];
        d1[i] = s * dsigmoid(a1[i]);
    }

    for (int i = 0; i < H2; i++)
        for (int j = 0; j < OUTPUT; j++)
            W3[i][j] -= LR * d3[j] * a2[i];
    for (int i = 0; i < OUTPUT; i++)
        B3[i] -= LR * d3[i];

    for (int i = 0; i < H1; i++)
        for (int j = 0; j < H2; j++)
            W2[i][j] -= LR * d2[j] * a1[i];
    for (int i = 0; i < H2; i++)
        B2[i] -= LR * d2[i];

    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < H1; j++)
            W1[i][j] -= LR * d1[j] * x[i];
    for (int i = 0; i < H1; i++)
        B1[i] -= LR * d1[i];
}

int argmax(float v[OUTPUT]) {
    int m = 0;
    for (int i = 1; i < OUTPUT; i++)
        if (v[i] > v[m]) m = i;
    return m;
}

void print_digit(float x[INPUT]) {
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            float v = x[r * 28 + c];
            if (v > 0.7) printf("@");
            else if (v > 0.3) printf("+");
            else printf(".");
        }
        printf("\n");
    }
}

int main() {
    srand(1);

    printf("Loading the dataset csv\n");
    load_mnist("../mnist_train.csv", X_train, Y_train, TRAIN_SAMPLES);
    load_mnist("../mnist_test.csv",  X_test,  Y_test,  TEST_SAMPLES);

    for (int i = 0; i < INPUT; i++)
        for (int j = 0; j < H1; j++)
            W1[i][j] = randf();

    for (int i = 0; i < H1; i++)
        for (int j = 0; j < H2; j++)
            W2[i][j] = randf();

    for (int i = 0; i < H2; i++)
        for (int j = 0; j < OUTPUT; j++)
            W3[i][j] = randf();

    printf("Training started\n");

    for (int e = 0; e < EPOCHS; e++) {
        float loss = 0;
        int correct = 0;

        for (int i = 0; i < TRAIN_SAMPLES; i++) {
            float a1[H1], a2[H2], a3[OUTPUT];
            forward(X_train[i], a1, a2, a3);

            int p = argmax(a3);
            if (p == Y_train[i]) correct++;

            for (int k = 0; k < OUTPUT; k++) {
                float t = (k == Y_train[i]);
                float d = a3[k] - t;
                loss += 0.5f * d * d;
            }

            backward(X_train[i], Y_train[i], a1, a2, a3);
        }

        printf("Epoch %d | loss=%.4f | acc=%.2f%%\n",
               e + 1,
               loss / TRAIN_SAMPLES,
               100.0f * correct / TRAIN_SAMPLES);
    }

    printf("\nRandom test samples:\n");

    for (int k = 0; k < 3; k++) {
        int idx = rand() % TEST_SAMPLES;
        float a1[H1], a2[H2], a3[OUTPUT];

        forward(X_test[idx], a1, a2, a3);

        printf("\nTrue: %d  Pred: %d\n",
               Y_test[idx], argmax(a3));

        print_digit(X_test[idx]);
    }

    return 0;
}
