#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float random_float(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

void weight_bias_initialization(
    float w_ih[2][2],
    float w_ho[2][1],
    float b_h[2],
    float b_o[1]
) {
    float limit = 2.4f / 2.0f;   

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            w_ih[i][j] = random_float(-limit, limit);

    for (int i = 0; i < 2; i++)
        w_ho[i][0] = random_float(-limit, limit);

    for (int i = 0; i < 2; i++)
        b_h[i] = random_float(-limit, limit);

    b_o[0] = random_float(-limit, limit);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}


float loss(float y_pred, float y_true) {
    float d = y_pred - y_true;
    return 0.5f * d * d;
}

void forward_propagation(
    int x[2],
    float w_ih[2][2],
    float w_ho[2][1],
    float b_h[2],
    float b_o[1],
    float h[2],
    float *y
) {
    for (int j = 0; j < 2; j++) {
        float sum = 0.0f;
        for (int i = 0; i < 2; i++)
            sum += x[i] * w_ih[i][j];
        sum += b_h[j];
        h[j] = sigmoid(sum);
    }

    float sum = 0.0f;
    for (int j = 0; j < 2; j++)
        sum += h[j] * w_ho[j][0];
    sum += b_o[0];
    *y = sigmoid(sum);
}

void backward_propagation(
    int x[2],
    int target,
    float w_ih[2][2],
    float w_ho[2][1],
    float b_h[2],
    float b_o[1],
    float h[2],
    float y,
    float lr
) {
    float delta_o = (y - target) * y * (1.0f - y);

    float delta_h[2];
    for (int j = 0; j < 2; j++)
        delta_h[j] = (w_ho[j][0] * delta_o) * h[j] * (1.0f - h[j]);

    for (int j = 0; j < 2; j++)
        w_ho[j][0] -= lr * delta_o * h[j];
    b_o[0] -= lr * delta_o;

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++)
            w_ih[i][j] -= lr * delta_h[j] * x[i];
        b_h[j] -= lr * delta_h[j];
    }
}


int main() {

    srand(time(NULL));

    int input[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    int target[4] = {0, 1, 1, 0};

    int epochs;
    printf("Enter number of epochs: ");
    scanf("%d", &epochs);

    float lr = 0.1f;

    float w_ih[2][2];
    float w_ho[2][1];
    float b_h[2];
    float b_o[1];

    float h[2];
    float y;

    weight_bias_initialization(w_ih, w_ho, b_h, b_o);

    for (int e = 0; e < epochs; e++) {
        float total_loss = 0.0f;

        for (int i = 0; i < 4; i++) {
            forward_propagation(input[i], w_ih, w_ho, b_h, b_o, h, &y);
            total_loss += loss(y, target[i]);
            backward_propagation(input[i], target[i],
                                  w_ih, w_ho, b_h, b_o,
                                  h, y, lr);
        }

        if (e % 1000 == 0)
            printf("Epoch %d | Loss %.6f\n", e, total_loss);
    }

    printf("\nFinal XOR predictions:\n");
    for (int i = 0; i < 4; i++) {
        forward_propagation(input[i], w_ih, w_ho, b_h, b_o, h, &y);
        printf("(%d,%d) -> %.4f\n", input[i][0], input[i][1], y);
    }

    return 0;
}
