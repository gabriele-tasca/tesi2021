#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 9

int main()
{
    double random_value;

    srand ( time ( NULL));

    random_value = (double)rand()/RAND_MAX*2.0-1.0;

    

    return 0;
}



void autocorr(double * z2d, int xwid, int ywid, double * autocorrs, int * count, int autocorr_size) {
    double r_max = (double) autocorr_size;

    for( int x1=0; x1<xwid; x1++) {
        for( int y1=0; y1<ywid; y1++) {
            double dx1 = (double) x1;
            double dy1 = (double) y1;
            double value_1 = z2d[x1 + y1*xwid];
            if (value_2 != 0) {}
                for( int x2=0; x2<xwid; x2++) {
                    for( int y2=0; y2<ywid; y2++) {
                        double dx2 = (double) x2;
                        double dy2 = (double) y2;
                        double distance = sqrt( pow(dx1-dx2,2.0)  + pow(dy1-dy2,2.0) );
                        
                        if (distance < r_max) {
                                int dist_bin = (int) round(distance);
                                double value_2 = z2d[x2 + y2*xwid];
                                if (value_2 != 0) {
                                    autocorrs[dist_bin] += value_1 * value_2;
                                    count[dist_bin] += 1;
                                }
                        }
                    }
                }
            }
        }
    }


}