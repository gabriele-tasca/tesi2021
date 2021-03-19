#include <math.h>

int power_of_two(int n) {
    return 2<<(n-1);
}

void fbm2D(double H, int N, double base_stdev, double * out_array) {
    int side = power_of_two(N) + 1;
    for (int g=0; g<N; g++) {
        int gpow = power_of_two(g);
        gstep = power_of_two(N-g);

        double stdev = pow(0.5, (H* (float) g)*base_stdev);


    }

}