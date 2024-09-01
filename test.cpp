#include <iostream>
#include <immintrin.h>

void saxpy(int n, float a, const float* x, float* y) {
    __m128 a_vec = _mm_set1_ps(a);
    for (int i = 0; i <= n - 4; i += 4) {
        __m128 x_vec = _mm_loadu_ps(&x[i]);
        __m128 y_vec = _mm_loadu_ps(&y[i]);
        y_vec = _mm_add_ps(_mm_mul_ps(a_vec, x_vec), y_vec);
        _mm_storeu_ps(&y[i], y_vec);
    }
    for (int i = n - (n % 4); i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    int n = 5;
    float a = 2.0f;
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    saxpy(n, a, x, y);

    for (int i = 0; i < n; ++i) {
        std::cout << y[i] << " ";
    }

    return 0;
}
