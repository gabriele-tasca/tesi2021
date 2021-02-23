import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt

import fract


def profile_var_range_from_z2d_corrected_corners(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales_flucts = np.zeros((s_max - s_min ,2))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s

        index = 0
        miss_count = 0

        discarded_rows = n%s 
        discarded_cols = m%s

        disc_treshold = 10

        corners = [ (0,0) ]
        if (discarded_rows > disc_treshold):
            corners.append( (0,1) )
        if (discarded_cols > disc_treshold):
            corners.append( (1,0) )
        if (discarded_rows > disc_treshold and discarded_cols > disc_treshold):
            corners.append( (1,1))

        segment_fs = np.zeros(n_submat_m*n_submat_n * len(corners))

        for corn in corners:
            r_offset = corn[0] * discarded_rows
            c_offset = corn[1] * discarded_cols
            for v in range(0, n_submat_m):
                for w in range(0, n_submat_n):
                    # slice out segment from 2d array
                    z_segment = (z2d[r_offset+s*v:r_offset+s*(v+1), c_offset+s*w:c_offset+s*(w+1)])

                    if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                        f = z_segment.max() - z_segment.min()


                        segment_fs[index] = f
                        index += 1
                    else:
                        miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        av_range = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = av_range
        if messages == True:
            print( "av_range", (av_range), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]



def profile_var_range_from_z2d_corrected(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales_flucts = np.zeros((s_max - s_min ,2))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s

        index = 0
        miss_count = 0

        discarded_rows = n%s 
        discarded_cols = m%s 

        segment_fs = np.zeros(n_submat_m*n_submat_n* discarded_rows * discarded_cols)

        for r_offset in range(0, discarded_rows):
            for c_offset in range(0, discarded_cols):
                for v in range(0, n_submat_m):
                    for w in range(0, n_submat_n):
                        # slice out segment from 2d array
                        z_segment = (z2d[r_offset+s*v:r_offset+s*(v+1), c_offset+s*w:c_offset+s*(w+1)])

                        if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                            f = z_segment.max() - z_segment.min()


                            segment_fs[index] = f
                            index += 1
                        else:
                            miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        av_range = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = av_range
        if messages == True:
            print( "av_range", (av_range), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]



def profile_var_range_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales_flucts = np.zeros((s_max - s_min ,2))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s
        segment_fs = np.zeros(n_submat_m*n_submat_n)

        index = 0
        miss_count = 0

        for v in range(0, n_submat_m):
            for w in range(0, n_submat_n):
                # slice out segment from 2d array
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                    f = z_segment.max() - z_segment.min()


                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        av_range = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = av_range
        if messages == True:
            print( "av_range", (av_range), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]



def profile_var_rms_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales_flucts = np.zeros((s_max - s_min ,2))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s
        segment_fs = np.zeros(n_submat_m*n_submat_n)

        index = 0
        miss_count = 0

        for v in range(0, n_submat_m):
            for w in range(0, n_submat_n):
                # slice out segment from 2d array
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                    f = np.std(z_segment.ravel())

                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        av_range = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = av_range
        if messages == True:
            print( "av_range", (av_range), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]



def profile_var_RS_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales_flucts = np.zeros((s_max - s_min ,2))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s
        segment_fs = np.zeros(n_submat_m*n_submat_n)

        index = 0
        miss_count = 0

        for v in range(0, n_submat_m):
            for w in range(0, n_submat_n):
                # slice out segment from 2d array
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                    f = (z_segment.max() - z_segment.min()) / ( np.std(z_segment.ravel()) )

                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        av_range = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = av_range
        if messages == True:
            print( "av_range", (av_range), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]

# plot

def pow_law(x, a, b):
    return a * x**b


if __name__ == '__main__':
    ex_z2d = fract.fbm2D(0.6, 10)

    H, c, scales, flucts = profile_var_range_from_z2d_corrected_corners(ex_z2d, messages=False)


    popt, pcov = scipy.optimize.curve_fit(pow_law, scales, flucts)
    # popt, pcov = scipy.optimize.curve_fit(pow_law, scales, flucts, sigma=flucts)
    
    popt

    for i in range(4,20):
        plt.axvline( (ex_z2d.shape[0])//i )

    plt.plot(scales, pow_law(scales, *popt), color="springgreen")

    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (10**c)*scales**H, color="purple")

    plt.title("H = " + str(H))
    plt.show()

    for i in range(4,10):
        plt.axvline( (ex_z2d.shape[0])//i )
    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (10**c)*scales**H, color="purple")
    plt.plot(scales, pow_law(scales, *popt), color="springgreen")

    plt.xscale("log")
    plt.yscale("log")
    plt.show()


