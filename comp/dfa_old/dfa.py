import numpy as np
import scipy.stats 
import matplotlib.pyplot as plt

# %matplotlib tk

def segment_fluct2(x, y, z, lin_s):
    design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

    param = np.array(np.linalg.lstsq(design, z)[0])
    trend_z = design @ param

    z_residue = z - trend_z
    fluct2 = np.sum(z_residue**2)/(lin_s**2)
    return fluct2

def segment_fluct2_plot(x, y, z, lin_s):
    design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

    param = np.array(np.linalg.lstsq(design, z)[0])
    trend_z = design @ param

    z_residue = z - trend_z
    fluct2 = np.sum(z_residue**2)/(lin_s**2)
    return fluct2, trend_z

# plotting single segment regression
# z2d = np.load("data-4.npy") 
# m,n = z2d.shape
# x2d,y2d = np.mgrid[:m,:n]
# x = x2d.ravel()
# y = y2d.ravel()
# z = z2d.ravel()

# fluct, trend_z = segment_fluct2_plot(x,y,z,n)

# trend_z2d = np.reshape(trend_z, (-1,n))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(x,y,z)
# ax.plot_surface(x2d,y2d,trend_z2d, color=(0,0,1,0.5))
# # ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2])
# plt.show()





def square_grid_dfa(x, y, z, s_min = 6, s_max = "auto"):
    n = np.sqrt(x.size).astype(int)
    # x2d,y2d = np.mgrid[:n,:n]
    x2d = x.reshape((n,n))
    y2d = y.reshape((n,n))
    z2d = z.reshape((n,n))


    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales = np.zeros((s_max - s_min))
    flucts = np.zeros((s_max - s_min))

    for j,s in enumerate(range(s_min,s_max)):
        m = n//s
        segment_fs = np.zeros(m**2)
        index = 0
        for v in range(0, m):
            for w in range(0, m):
                # slice out segment from 2d array 
                x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
                y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])
                f = segment_fluct2(x_segment.ravel(), y_segment.ravel(), z_segment.ravel(), s)
                segment_fs[index] = f
                index += 1
        fluct_av_sq = np.sum(segment_fs)/(m**2)
        scales[j] = s
        flucts[j] = np.sqrt(fluct_av_sq)

    s_log = np.log10(scales)
    f_log = np.log10(flucts)

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)

    return H, c, scales, flucts

def square_grid_dfa_from_z2d(z2d, s_min = 6, s_max = "auto"):
    ex_m,ex_n = z2d.shape
    ex_x2d,ex_y2d = np.mgrid[:ex_m,:ex_n]
    ex_x = ex_x2d.ravel()
    ex_y = ex_y2d.ravel()
    ex_z = z2d.ravel()
    H, c, scales, flucts = square_grid_dfa(ex_x, ex_y, ex_z, s_min, s_max)
    return H, c, scales, flucts


def profile_dfa(x, y, z, s_min = 6, s_max = "auto", min_nonzero = 0.9):
    n = np.sqrt(x.size).astype(int)
    # x2d,y2d = np.mgrid[:n,:n]
    x2d = x.reshape((n,n))
    y2d = y.reshape((n,n))
    z2d = z.reshape((n,n))

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max

    scales = np.zeros((s_max - s_min))
    flucts = np.zeros((s_max - s_min))

    for j,s in enumerate(range(s_min,s_max)):
        m = n//s
        segment_fs = np.zeros(m**2)

        index = 0
        miss_count = 0

        for v in range(0, m):
            for w in range(0, m):
                # slice out segment from 2d array 
                x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
                y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):
                    f = segment_fluct2(x_segment.ravel(), y_segment.ravel(), z_segment.ravel(), s)
                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1
        # remove unfilled slots from tail
        # print(miss_count/(index+miss_count))
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        fluct_av_sq = np.sum(segment_fs)/(index)
        scales[j] = s
        flucts[j] = np.sqrt(fluct_av_sq)

    s_log = np.log10(scales)
    f_log = np.log10(flucts)

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    print("H =  ",H)
    return H, c, scales, flucts


def profile_dfa_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9):
    ex_m,ex_n = z2d.shape
    ex_x2d,ex_y2d = np.mgrid[:ex_m,:ex_n]
    ex_x = ex_x2d.ravel()
    ex_y = ex_y2d.ravel()
    ex_z = z2d.ravel()
    H, c, scales, flucts = profile_dfa(ex_x, ex_y, ex_z, s_min, s_max, min_nonzero = min_nonzero)
    return H, c, scales, flucts




def profile_dfa_from_z2d_2(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
    (m, n) = z2d.shape
    x2d,y2d = np.mgrid[:m,:n]

    s_min = 6
    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max


    # scales - flucts table
    scales_flucts = np.zeros((s_max - s_min ,2))
    # scales = np.zeros((s_max - s_min))
    # flucts = np.zeros((s_max - s_min))

    for j,s in enumerate(range(s_min,s_max)):
        n_submat_n = n//s
        n_submat_m = m//s
        segment_fs = np.zeros(n_submat_m*n_submat_n)

        index = 0
        miss_count = 0

        for v in range(0, n_submat_m):
            for w in range(0, n_submat_n):
                # slice out segment from 2d array
                x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
                y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):
                    f = segment_fluct2(x_segment.ravel(), y_segment.ravel(), z_segment.ravel(), s)
                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        fluct_av_sq = np.sum(segment_fs)/(index)
        scales_flucts[j,0] = s
        scales_flucts[j,1] = np.sqrt(fluct_av_sq)
        if messages == True:
            print( "fluct_av_sq", (fluct_av_sq), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")

    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]

    
    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales_flucts[:,0])
    f_log = np.log10(scales_flucts[:,1])    

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(s_log, f_log)
    # print("H =  ",H)
    return H, c, scales_flucts[:,0], scales_flucts[:,1]




def pow_law(x, exp, coeff):
    return coeff * x**exp

if __name__ == '__main__':
    # plot
    import fract

    ex_z2d = fract.fbm2D(0.3, 10)

    H, c, scales, flucts = fract.profile_dfa_from_z2d(ex_z2d)

    popt, pcov = scipy.optimize.curve_fit(pow_law, scales, flucts)
    
    popt

    for i in range(4,10):
        plt.axvline( (ex_z2d.shape[0])//i )

    plt.plot(scales, pow_law(scales, *popt), color="green")

    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (10**c)*scales**H, color="purple")
    plt.title("H = " + str(H))
    plt.show()


    plt.plot(scales, pow_law(scales, *popt), color="green")

    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (10**c)*scales**H, color="purple")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("H = " + str(H))
    plt.show()
