import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt


def fbm2D(H, N=4, base_stdev=1.0):
    side = 2**N + 1
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )
    for g in range(1,N+1):
        gpow = 2**g
        gstep = (side-1) // gpow
        stdev = (0.5)**(H*g) * base_stdev

        # random additions
        rands = rng.normal( size=(gpow+1,gpow+1), scale=stdev)
        vec[  0:side:gstep  ,  0:side:gstep  ] += rands

        # interpolation
        if g < N:
            half = gstep//2
            # interpolation 1
            north_west = vec[  0:side-gstep:gstep, 0:side-gstep:gstep  ]
            north_east = vec[  0:side-gstep:gstep, gstep:side:gstep  ]
            south_west = vec[  gstep:side:gstep, 0:side-gstep:gstep  ]
            south_east = vec[  gstep:side:gstep, gstep:side:gstep  ]
            
            vec[  half:side:gstep, half:side:gstep  ] = (north_west + north_east + south_west + south_east)/4

            ##### interpolation 2.1

            # #   left edge
            # edge_up_1l = vec[  0:side-gstep:gstep, 0  ]
            # edge_down_1l = vec[  gstep:side+half:gstep, 0  ]
            # edge_right_1l = vec[ half:side:gstep, half]
            # vec[  half:side:gstep, 0  ] = (edge_up_1l + edge_down_1l + edge_right_1l)/3
            
            # #  right edge
            # edge_up_1r = vec[  0:side-gstep:gstep, side-1 ]
            # edge_down_1r = vec[  gstep:side+half:gstep, side-1  ]
            # edge_left_1r = vec[ half:side:gstep, side-1-half ]
            # vec[  half:side:gstep, side-1  ] = (edge_up_1r + edge_down_1r + edge_left_1r)/3


            #   bulk
            up3 = vec[  0:side-half:gstep, gstep:side-gstep:gstep  ]
            down3 = vec[  2*half:side+half:gstep, gstep:side-gstep:gstep  ]
            left3 = vec[ half:side:gstep, half:side-half-gstep:gstep]
            right3 = vec[ half:side:gstep, half+gstep:side-half:gstep]
            vec[  half:side:gstep, gstep:side-gstep:gstep  ] = (up3 + down3 + left3 + right3)/4
            
            ##### interpolation 2.2

            # # up edge
            # edge_left_2u = vec[ 0, 0:side-half:gstep ]
            # edge_right_2u = vec[  0, half+half:side+half:gstep  ]
            # edge_down_2u = vec[ half, half:side:gstep]
            # vec[  0, half:side:gstep  ] = (edge_left_2u + edge_right_2u + edge_down_2u)/3
            
            # # down edge
            # edge_left_2d = vec[ side-1, 0:side-half:gstep ]
            # edge_right_2d = vec[ side-1, half+half:side+half:gstep  ]
            # edge_up_2d = vec[ side-half, half:side:gstep]
            # vec[  0, half:side:gstep  ] = (edge_left_2d + edge_right_2d + edge_up_2d)/3

            # bulk
            left4 = vec[  gstep:side-gstep:gstep, 0:side-half:gstep ]
            right4 = vec[  gstep:side-gstep:gstep, half+half:side+half:gstep  ]
            up4 = vec[ half:side-half-gstep:gstep, half:side:gstep]
            down4 = vec[ half+gstep:side-half:gstep, half:side:gstep]

            vec[  gstep:side-gstep:gstep, half:side:gstep  ] = (up4 + down4 + left4 + right4)/4


    return vec[1:side-1,1:side-1]




def fbm2D_old(H, N=4, base_stdev=1.0):
    side = 2**N + 1
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )
    for g in range(1,N+1):
        gpow = 2**g
        gstep = (side-1) // gpow
        stdev = (0.5)**(H*g) * base_stdev

        # random additions
        rands = rng.normal( size=(gpow+1,gpow+1), scale=stdev)
        vec[  0:side:gstep  ,  0:side:gstep  ] += rands

        # interpolation
        if g != N:
            delta = gstep//2
            up = vec[  0:side-delta:gstep, 0:side:gstep  ]
            down = vec[  0+2*delta:side+delta:gstep, 0:side:gstep  ]
            vec[  0+delta:side:gstep, 0:side:gstep  ] = (up + down)/2

            left = vec[  0:side:gstep, 0:side-delta:gstep]
            right = vec[  0:side:gstep,  0+2*delta:side+delta:gstep]
            vec[  0:side:gstep, 0+delta:side:gstep  ] = (left + right)/2

            
            up2 = vec[  0:side-delta:gstep, 0+delta:side:gstep  ] 
            down2 = vec[  0+2*delta:side:gstep, 0+delta:side:gstep  ]

            left2 = vec[  0+delta:side:gstep, 0:side-delta:gstep  ]
            right2 = vec[  0+delta:side:gstep, 0+2*delta:side:gstep  ]
            vec[  0+delta:side:gstep, 0+delta:side:gstep  ] = (up2 + down2 + left2 + right2)/4

    return vec

###########


def power_law(x, exp, coeff):
    return coeff * x**(exp)

def weighted_power_law_fit(xdata, ydata, sigmas, p0=(1,3)):
    popt, pcov = scipy.optimize.curve_fit(power_law, ydata, powers, sigma=sigmas, p0=pzero)
    return popt, pcov

def linear_log_fit(xdata, ydata):
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]
  
    # s_f_log = np.log10(scales_flucts)
    x_log = np.log10(xdata)
    y_log = np.log10(ydata)

    # A = np.vstack([np.ones(len(s_log)), s_log]).T
    H, c, r_value, p_value, std_err = scipy.stats.linregress(x_log, y_log)
    # print("H =  ",H)
    return H, 10**c


########### DFA 


def segment_fluct2(x, y, z, lin_s):
    design = np.column_stack((x*x, y*y, x*y, x, y, np.full(lin_s**2, 1.0)))

    param = np.array(np.linalg.lstsq(design, z)[0])
    trend_z = design @ param

    z_residue = z - trend_z
    fluct2 = np.sum(z_residue**2)/(lin_s**2)
    return fluct2

def profile_dfa_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, messages = False):
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



########## variable bandwidth methods

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



########## fourier methods



def profile_fourier_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, images=False, corr=False):
    side = z2d.shape[0]
    
    if corr:
        corr2d = fftconvolve(z2d, z2d[::-1,::-1], mode="full")
        if images:
            plt.imshow(corr2d, interpolation="None")
            plt.show()

    image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d))   ) **2
    
    if images:
        plt.imshow(z2d, interpolation="None")
        plt.show()
        plt.imshow(np.log(image_fft), interpolation="None")
        plt.show()

    sx, sy = image_fft.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    distances = np.sqrt( ( X - sx/2)**2 + (Y - sy/2)**2  )

    # freq_points = int(side /np.sqrt(2) )
    freq_points = int(side /2 )
    freqs = range(0,freq_points)
    powers = np.zeros( freq_points )
    for r in range(0,freq_points):
        powers[r] = np.mean( image_fft[  (distances > (r-0.5) ) & (distances <= r+0.5)  ] )


#     freq_log = np.log10(freqs)
#     pow_log = np.log10(powers)

#    freq_exp, c, r_value, p_value, std_err = scipy.stats.linregress(freq_log[1:], pow_log[1:])

    (freq_exp, c), pcov = weighted_frequence_power_law_fit(xdata=freqs[1:], ydata=powers[1:], sigmas=powers[1:])

    detect_H = -(freq_exp + 1.5)/2
    return detect_H, freq_exp, c, freqs, powers
