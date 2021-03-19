#include <stdlib.h>
#include <stdio.h>

#define EMPTY 0.0
#define EMPTYBUF -777.777

void dma(double * z2d, int M, int N, double * flucts_out, int s_min, int s_max, double min_nonzero) {

    int n_scales = (s_max - s_min);

    // buffer to store partial means and stuff. note that some parts 
    //    of the buffer are never even initialized.
    double * buf = malloc(sizeof *buf *M*N);
    // loop on scales
    // first pass (s = s_min - 1): means are calculated from scratch.
    // all but 1 of these means can be still optimized.
    int s = s_min -1;
    for (int v=s/2; v<M-(s-s/2); v++) {
        for (int w=s/2; w<N-(s-s/2); w++) {
            // check if center is EMPTY (outside perimeter), then 
            // check if the s_min x s_min tile goes outside the perimeter 
            if (z2d[v*N+w] != EMPTYBUF) {
                double res = 0;
                int non_empty_count = 0;
                for (int vv=-s/2; vv<(s-s/2); vv++) {
                    for (int ww=-s/2; ww<(s-s/2); ww++) {
                        int vvv = v + vv;
                        int www = w + ww;
                        if ( z2d[vvv*N + www] != EMPTY ) {
                            non_empty_count++;
                            res += z2d[vvv*N + www];
                        }
                    }
                }

                if (non_empty_count > min_nonzero*(double)(s*s)) {
                    buf[v*N+w] = res/((double) non_empty_count);
                } else {
                    buf[v*N+w] = EMPTYBUF;
                }

            } else {
                buf[v*N+w] = EMPTYBUF;
            }
        }
    }


    /////////////////////////////
    /////// DELETE THIS /////////
    /////////////////////////////
    // s_max = s_min + 50;
    /////////////////////////////

    int s_index = 0;
    for (int s=s_min; s<s_max; s++) {
        // grow averages
        // s even: grow bottom and right
        // s odd: grow top and left

        //
        //
        for (int v=s/2; v<M-(s-s/2); v++) { 
            for (int w=s/2; w<N-(s-s/2); w++) {
                if (buf[v*N+w] != EMPTYBUF) { 
                    double new_edges = 0;
                    int new_edges_miss = 0;
                    if (s%2 == 1) {
                        // s odd: grow bottom and right
                        for (int vv=-s/2; vv<(s-s/2); vv++) {
                            int vvv = v + vv;
                            int www = w + (s-s/2); 
                            if (z2d[vvv*N + www] != EMPTY ) {
                                new_edges += z2d[vvv*N + www];
                            } else {
                                new_edges_miss++;
                            }
                        }
                        // the -1 avoids counting the corner twice
                        for (int ww=-s/2; ww<(s-s/2)-1; ww++) {
                            int vvv = v + (s-s/2);
                            int www = w + ww; 
                            if (z2d[vvv*N + www] != EMPTY ) {
                                new_edges += z2d[vvv*N + www];
                            } else {
                                new_edges_miss++;
                            }
                        }
                    } else {
                        // s even: grow top and left
                        for (int vv=-s/2; vv<(s-s/2); vv++) {
                            int vvv = v + vv;
                            int www = w - s/2;
                            if (z2d[vvv*N + www] != EMPTY ) {
                                new_edges += z2d[vvv*N + www];
                            } else {
                                new_edges_miss++;
                            }
                        }
                        // the -1 avoids counting the corner twice
                        for (int ww=-s/2; ww<(s-s/2)-1; ww++) {
                            int vvv = v - s/2;
                            int www = w + ww;
                            if (z2d[vvv*N + www] != EMPTY ) {
                                new_edges += z2d[vvv*N + www];
                            } else {
                                new_edges_miss++;
                            }
                        }
                    }

                    if (new_edges_miss < 3) {
                        buf[v*N+w] = buf[v*N+w]*((double)(s-1)*(s-1)) + new_edges;
                        buf[v*N+w] = buf[v*N+w]/((double)s*s);
                    } else {
                        buf[v*N+w] = EMPTYBUF;
                    }
                    new_edges_miss = 0;
                }
            }
        }
        // collect average fluctuation
        double fluct_sq_av = 0.0;
        double p_count = 0.0;
        double empty_count = 0.0;

        double overflow_fix = 1.0;
        if ( s < 100.0 ) {
            overflow_fix = 1.0;
        }

        for (int v=s/2; v<M-(s-s/2); v++) { 
            for (int w=s/2; w<N-(s-s/2); w++) {
                if (buf[v*N+w] != EMPTYBUF) {
                    double fluct_sq = (z2d[v*N+w] - buf[v*N+w]);
                    fluct_sq_av += (fluct_sq*fluct_sq)*overflow_fix;
                    p_count += 1.0;
                } else {
                    empty_count = empty_count +1.0;
                }
            }
        }
        if (p_count >= 10) {
            fluct_sq_av = fluct_sq_av/p_count;
            flucts_out[s_index] = (fluct_sq_av)/overflow_fix;

        } else {
            flucts_out[s_index] = -4000.0;
        }
        // scales_out[s_index] = s;
        s_index++;
    

    }

    // for (int v=s/2; v<M-(s-s/2); v++) {
    //     for (int w=s/2; w<N-(s-s/2); w++) {
    //         end_buf_out[v*N+w] = buf[v*N+w];
    //     }
    // }


}