import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.ndimage import mean as scipy_ndimage_mean


def fbm2D_spectral(H, N):
    half_beta = - (H + 1)

    n = 2**N

    rng = np.random.default_rng()
    synth_spectrum_quarter = rng.standard_normal((n//2,n//2)) + 1j * rng.standard_normal((n//2,n//2))

    tx = np.arange(0,n//2)
    ty = np.arange(0,n)

    radius = np.sqrt( ( tx.reshape((1,n//2))  )**2 + (ty.reshape((n,1)) - n/2 )**2  )
    radius = np.vstack(( radius[n//2:n, :], radius[0:n//2 , :] ))

    synth_spectrum = np.vstack( (synth_spectrum_quarter, synth_spectrum_quarter[::-1]) )

    # coeff = 4*np.pi* (tx)**(half_beta+2) /(half_beta+2)
    coeff = 10**(N/2)
    bell =  np.power(radius + 0.01, half_beta) * coeff

    filtered_synth_spectrum = synth_spectrum * bell

    synth_image = (np.fft.irfft2(filtered_synth_spectrum))
    synth_image = synth_image - np.mean(synth_image)
    return synth_image


def fbm2D_exact(H, N):
    n_out = 2**N
    rng = np.random.default_rng()

    alpha = 2*H

    #("set params")
    beta = 0
    c0 = 0
    c2 = 0
    R = 0
    n_real = 0
    if alpha <= 1.5: 
        R = 1
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
        n_real = n_out
    else: 
        R = 2
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2*(R+2))/2
        c0 = beta*(R-1)**3+1-c2
        n_real = 2*n_out

    #("long loop")

    n = (np.ceil(n_real*np.sqrt(2))).astype(int)
    m = (np.ceil(n_real*np.sqrt(2))).astype(int)
    tx = (np.arange(0,n)/n)*R
    ty = (np.arange(0,m)/m)*R

    rows = np.zeros((m,n))

    rs = np.sqrt( ( tx.reshape((n,1)) )**2 + (ty.reshape((1,m)))**2  )

    rows[ rs <= 1] = c0-rs[rs<=1]**alpha+c2*rs[rs<=1]**2
    rows[ (rs > 1 )& (rs <= R)] = beta*(R-rs[(rs>1)&(rs<=R)])**3/rs[(rs>1)&(rs<=R)]

    # ^ this might be hard to read, but it's just a numpy-vectorized
    # application of this function to the "rows" array:
    # def rho(p1, p2):
    #     # create continuous isotropic function
    #     r = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    #     if r <= 1:
    #         out=c0-r**alpha+c2*r**2
    #     elif r<=R:
    #         out=beta*(R-r)**3/r
    #     else:
    #         out=0
    #     return out

    #("stacking")

    loc_a1 = rows
    loc_a2 = rows[:, -1:1:-1]
    loc_b1 = rows[-1:1:-1, :]
    loc_b2 = rows[-1:1:-1 , -1:1:-1]

    loc_c1 = np.hstack([loc_a1, loc_a2])
    loc_c2 = np.hstack([loc_b1, loc_b2])
    block_circ = np.vstack([loc_c1, loc_c2])

    #("eigenvalues")

    # compute eigen-values
    lam = np.real(np.fft.fft2(block_circ))/(4*(m-1)*(n-1))
    lam = np.sqrt(lam)
    Z1 = rng.standard_normal((2*(m-1),2*(n-1)))
    Z2 = rng.standard_normal((2*(m-1),2*(n-1)))

    #("second fft2")

    # generate field with covariance given by block circular matrix
    Z = Z1 + 1j*Z2
    F = np.fft.fft2(lam * Z)
    #("generate field ")

    F = F[0:m,0:n] # extract sub-block with desired covariance
    # (out,c0,c2) = rho( (0,0) ,(0,0),R,2*H)
    field1 = np.real(F)
    field2 = np.imag(F) # two independent fields
    field1 = field1 - field1[0,0] # set field zero at origin
    field2 = field2 - field2[0,0] # set field zero at origin

    #("corrections")

    # make correction for embedding with a term c2*r^2

    loc1 = ty.conj().T *rng.standard_normal()
    loc2 = tx*rng.standard_normal()
    loc3 = np.kron(loc1, loc2).reshape((loc1.size, loc2.size))
    field1 = field1 + loc3*np.sqrt(2*c2)
    field2 = field2 + loc3*np.sqrt(2*c2)


    if alpha <= 1.5: 
        n_cut = int(n/(np.sqrt(2)))
        m_cut = int(m/(np.sqrt(2)))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]
    else:
        n_cut = int(n/(np.sqrt(2)*2))
        m_cut = int(m/(np.sqrt(2)*2))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]

    return field1



def save_fbm2D_exact_generator_params(H, N):
    n_out = 2**N

    rng = np.random.default_rng()

    alpha = 2*H

    ###("set params")
    beta = 0
    c0 = 0
    c2 = 0
    R = 0
    n_real = 0
    if alpha <= 1.5: 
        R = 1
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
        n_real = n_out
    else: 
        R = 2
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2*(R+2))/2
        c0 = beta*(R-1)**3+1-c2
        n_real = 2*n_out

    ### experiment:
    # R = 2
    # beta = alpha*(2-alpha)/(3*R*(R**2-1))
    # c2 = (alpha-beta*(R-1)**2*(R+2))/2
    # c0 = beta*(R-1)**3+1-c2
    # n_real = 2*n_out

    #("long loop")

    n = (np.ceil(n_real*np.sqrt(2))).astype(int)
    m = (np.ceil(n_real*np.sqrt(2))).astype(int)
    tx = (np.arange(0,n)/n)*R
    ty = (np.arange(0,m)/m)*R

    rows = np.zeros((m,n))

    rs = np.sqrt( ( tx.reshape((n,1)) )**2 + (ty.reshape((1,m)))**2  )

    rows[ rs <= 1] = c0-rs[rs<=1]**alpha+c2*rs[rs<=1]**2
    rows[ (rs > 1 )& (rs <= R)] = beta*(R-rs[(rs>1)&(rs<=R)])**3/rs[(rs>1)&(rs<=R)]

    # ^ this might be hard to read, but it's just a numpy-vectorized
    # application of this function to the "rows" array:
    # def rho(p1, p2):
    #     # create continuous isotropic function
    #     r = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    #     if r <= 1:
    #         out=c0-r**alpha+c2*r**2
    #     elif r<=R:
    #         out=beta*(R-r)**3/r
    #     else:
    #         out=0
    #     return out

    #("stacking")

    loc_a1 = rows
    loc_a2 = rows[:, -1:1:-1]
    loc_b1 = rows[-1:1:-1, :]
    loc_b2 = rows[-1:1:-1 , -1:1:-1]

    loc_c1 = np.hstack([loc_a1, loc_a2])
    loc_c2 = np.hstack([loc_b1, loc_b2])
    block_circ = np.vstack([loc_c1, loc_c2])

    #("eigenvalues")

    # compute eigen-values
    lam = np.real(np.fft.fft2(block_circ))/(4*(m-1)*(n-1))
    lam = np.sqrt(lam)

    # return generator parameters

    return (lam, n,m,tx,ty,c2,alpha )




def fbm2D_exact_from_generator(lam, n,m,tx,ty,c2,alpha ):
    rng = np.random.default_rng()

    Z1 = rng.standard_normal((2*(m-1),2*(n-1)))
    Z2 = rng.standard_normal((2*(m-1),2*(n-1)))

    #("second fft2")

    # generate field with covariance given by block circular matrix
    Z = Z1 + 1j*Z2
    F = np.fft.fft2(lam * Z)
    #("generate field ")

    F = F[0:m,0:n] # extract sub-block with desired covariance
    # (out,c0,c2) = rho( (0,0) ,(0,0),R,2*H)
    field1 = np.real(F)
    field2 = np.imag(F) # two independent fields
    field1 = field1 - field1[0,0] # set field zero at origin
    field2 = field2 - field2[0,0] # set field zero at origin

    #("corrections")

    # make correction for embedding with a term c2*r^2

    loc1 = ty.conj().T *rng.standard_normal()
    loc2 = tx*rng.standard_normal()
    loc3 = np.kron(loc1, loc2).reshape((loc1.size, loc2.size))
    field1 = field1 + loc3*np.sqrt(2*c2)
    field2 = field2 + loc3*np.sqrt(2*c2)


    if alpha <= 1.5: 
        n_cut = int(n/(np.sqrt(2)))
        m_cut = int(m/(np.sqrt(2)))
        # n_cut = int(n/(np.sqrt(2)*2))
        # m_cut = int(m/(np.sqrt(2)*2))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]
    else:
        n_cut = int(n/(np.sqrt(2)*2))
        m_cut = int(m/(np.sqrt(2)*2))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]

    return (field1, field2)



##########
##########


def fbm2D_midpoint(H, N=4, base_stdev=1.0):
    side = 2**N + 1
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )

    # initialize corners
    stdev = (0.5)* base_stdev
    vec[  0  ,  0  ] += rng.normal( scale=stdev )
    vec[  0  ,  side-1  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  0  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  side-1  ] += rng.normal( scale=stdev )

    # loop 
    for g in range(1,N):
        # gpow = 2**g
        gstep = 2**(N-g)
        stdev = (0.5)**(H*g) * base_stdev

        gsize2 = 2**(g+1)+1

        half = gstep//2
        # diagonal interpolation
        north_west = vec[  0:side-gstep:gstep, 0:side-gstep:gstep  ]
        north_east = vec[  0:side-gstep:gstep, gstep:side:gstep  ]
        south_west = vec[  gstep:side:gstep, 0:side-gstep:gstep  ]
        south_east = vec[  gstep:side:gstep, gstep:side:gstep  ]
        
        vec[  half:side:gstep, half:side:gstep  ] = (north_west + north_east + south_west + south_east)/4

        ##### interpolation 2.1

        #   left edge
        edge_up_1l = vec[  0:side-gstep:gstep, 0  ]
        edge_down_1l = vec[  gstep:side+half:gstep, 0  ]
        edge_right_1l = vec[ half:side:gstep, half]
        vec[  half:side:gstep, 0  ] = (edge_up_1l + edge_down_1l + edge_right_1l)/3

        #  right edge
        edge_up_1r = vec[  0:side-gstep:gstep, side-1 ]
        edge_down_1r = vec[  gstep:side+half:gstep, side-1  ]
        edge_left_1r = vec[ half:side:gstep, side-1-half ]
        vec[  half:side:gstep, side-1  ] = (edge_up_1r + edge_down_1r + edge_left_1r)/3

        #   bulk
        up3 = vec[  0:side-half:gstep, gstep:side-gstep:gstep  ]
        down3 = vec[  2*half:side+half:gstep, gstep:side-gstep:gstep  ]
        left3 = vec[ half:side:gstep, half:side-half-gstep:gstep]
        right3 = vec[ half:side:gstep, half+gstep:side-half:gstep]
        vec[  half:side:gstep, gstep:side-gstep:gstep  ] = (up3 + down3 + left3 + right3)/4

        ##### interpolation 2.2

        # up edge
        edge_left_2u = vec[ 0, 0:side-half:gstep ]
        edge_right_2u = vec[  0, half+half:side+half:gstep  ]
        edge_down_2u = vec[ half, half:side:gstep]
        vec[  0, half:side:gstep  ] = (edge_left_2u + edge_right_2u + edge_down_2u)/3
        
        # down edge
        edge_left_2d = vec[ side-1, 0:side-half:gstep ]
        edge_right_2d = vec[ side-1, half+half:side+half:gstep  ]
        edge_up_2d = vec[ side-half, half:side:gstep]
        vec[  side-1, half:side:gstep  ] = (edge_left_2d + edge_right_2d + edge_up_2d)/3

        # bulk
        left4 = vec[  gstep:side-gstep:gstep, 0:side-half:gstep ]
        right4 = vec[  gstep:side-gstep:gstep, half+half:side+half:gstep  ]
        up4 = vec[ half:side-half-gstep:gstep, half:side:gstep]
        down4 = vec[ half+gstep:side-half:gstep, half:side:gstep]

        vec[ gstep:side-gstep:gstep, half:side:gstep ] = (up4 + down4 + left4 + right4)/4


        # random addition (sq 1 + sq 2 bulk)
        rands = rng.normal( size=(gsize2,gsize2), scale=stdev*np.sqrt(2))
        vec[  0:side:half, 0:side:half  ] += rands


    # return vec[1:side-1,1:side-1]
    return vec


def fbm2D_midpoint_old2(H, N=4, base_stdev=1.0):
    side = 2**N + 1
    rng = np.random.default_rng()

    vec = np.zeros( (side,side) )

    # initialize corners
    stdev = (0.5)* base_stdev
    vec[  0  ,  0  ] += rng.normal( scale=stdev )
    vec[  0  ,  side-1  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  0  ] += rng.normal( scale=stdev )
    vec[  side-1  ,  side-1  ] += rng.normal( scale=stdev )

    # loop 
    for g in range(1,N):
        gpow = 2**g
        gstep = 2**(N-g)
        stdev = (0.5)**(H*g) * base_stdev


        half = gstep//2
        # diagonal interpolation
        north_west = vec[  0:side-gstep:gstep, 0:side-gstep:gstep  ]
        north_east = vec[  0:side-gstep:gstep, gstep:side:gstep  ]
        south_west = vec[  gstep:side:gstep, 0:side-gstep:gstep  ]
        south_east = vec[  gstep:side:gstep, gstep:side:gstep  ]
        
        vec[  half:side:gstep, half:side:gstep  ] = (north_west + north_east + south_west + south_east)/4

        # random addition (diag)
        rands = rng.normal( size=(gpow,gpow), scale=stdev*np.sqrt(2))
        vec[  half:side:gstep, half:side:gstep  ] += rands

        ##### interpolation 2.1

        #   left edge
        edge_up_1l = vec[  0:side-gstep:gstep, 0  ]
        edge_down_1l = vec[  gstep:side+half:gstep, 0  ]
        edge_right_1l = vec[ half:side:gstep, half]
        vec[  half:side:gstep, 0  ] = (edge_up_1l + edge_down_1l + edge_right_1l)/3
        # random addition (sq 1 left edge)
        rands = rng.normal( size=vec[half:side:gstep, 0].shape, scale=stdev)
        vec[  half:side:gstep, 0  ] += rands

        #  right edge
        edge_up_1r = vec[  0:side-gstep:gstep, side-1 ]
        edge_down_1r = vec[  gstep:side+half:gstep, side-1  ]
        edge_left_1r = vec[ half:side:gstep, side-1-half ]
        vec[  half:side:gstep, side-1  ] = (edge_up_1r + edge_down_1r + edge_left_1r)/3
        # random addition (sq 1 right edge)
        rands = rng.normal( size=vec[half:side:gstep, side-1].shape, scale=stdev)
        vec[  half:side:gstep, side-1  ] += rands

        #   bulk
        up3 = vec[  0:side-half:gstep, gstep:side-gstep:gstep  ]
        down3 = vec[  2*half:side+half:gstep, gstep:side-gstep:gstep  ]
        left3 = vec[ half:side:gstep, half:side-half-gstep:gstep]
        right3 = vec[ half:side:gstep, half+gstep:side-half:gstep]
        vec[  half:side:gstep, gstep:side-gstep:gstep  ] = (up3 + down3 + left3 + right3)/4
        
        # random addition (sq 1 bulk)
        rands = rng.normal( size=(gpow,gpow-1), scale=stdev)
        vec[  half:side:gstep, gstep:side-gstep:gstep  ] += rands

        ##### interpolation 2.2

        # up edge
        edge_left_2u = vec[ 0, 0:side-half:gstep ]
        edge_right_2u = vec[  0, half+half:side+half:gstep  ]
        edge_down_2u = vec[ half, half:side:gstep]
        vec[  0, half:side:gstep  ] = (edge_left_2u + edge_right_2u + edge_down_2u)/3
        # random addition (sq 2 up edge)
        rands = rng.normal( size=vec[0, half:side:gstep].shape, scale=stdev)
        vec[  0, half:side:gstep  ] += rands
        
        # down edge
        edge_left_2d = vec[ side-1, 0:side-half:gstep ]
        edge_right_2d = vec[ side-1, half+half:side+half:gstep  ]
        edge_up_2d = vec[ side-half, half:side:gstep]
        vec[  side-1, half:side:gstep  ] = (edge_left_2d + edge_right_2d + edge_up_2d)/3
        # random addition (sq 2 up edge)
        rands = rng.normal( size=vec[side-1, half:side:gstep].shape, scale=stdev)
        vec[  side-1, half:side:gstep  ] += rands

        # bulk
        left4 = vec[  gstep:side-gstep:gstep, 0:side-half:gstep ]
        right4 = vec[  gstep:side-gstep:gstep, half+half:side+half:gstep  ]
        up4 = vec[ half:side-half-gstep:gstep, half:side:gstep]
        down4 = vec[ half+gstep:side-half:gstep, half:side:gstep]

        vec[ gstep:side-gstep:gstep, half:side:gstep ] = (up4 + down4 + left4 + right4)/4

        # random addition (sq 2)
        rands = rng.normal( size=(gpow-1,gpow), scale=stdev)
        vec[ gstep:side-gstep:gstep, half:side:gstep ] += rands

    # return vec[1:side-1,1:side-1]
    return vec



# z2d = fbm2D(H=0.6, N=10)
# plt.imshow(z2d)
# plt.show()



def cut_profile(image, contour_H, treshold=0.0, invert=1):
    n, m = image.shape
    side = max(n,m)
    N = np.ceil(np.log2(side-1)).astype(int)

    contour_im = fbm2D_midpoint(contour_H, N=N)
    # this works because midpoint is sligthly bigger than the other 2. luckily it's also the fastest

    contour_im = contour_im[0:n,0:m]
    contour_im[ invert*contour_im >= invert*treshold ] = 0.0
    contour_im[ invert*contour_im < invert*treshold ] = 1.0
    return np.multiply(contour_im, image)

###########


def power_law_const(x, exp, coeff, const):
    return coeff * x**(exp) + const


def power_law(x, exp, coeff):
    return coeff * x**(exp)

def weighted_power_law_fit_const(xdata, ydata, sigmas, p0=(1,3,0)):
    popt, pcov = scipy.optimize.curve_fit(power_law_const, xdata, ydata, sigma=sigmas, p0=p0)
    return popt[0], popt[1], popt[2], pcov

def weighted_power_law_fit(xdata, ydata, sigmas, p0=(1,3)):
    popt, pcov = scipy.optimize.curve_fit(power_law, xdata, ydata, sigma=sigmas, p0=p0)
    return popt[0], popt[1], pcov


def linear_log_fit(xdata, ydata):

    xdata_log = np.log10(xdata)
    ydata_log = np.log10(ydata)

    exp, c, r_value, p_value, std_err = scipy.stats.linregress(xdata_log, ydata_log)
    return exp, 10**c

def autoseeded_weighted_power_law_fit_const(xdata, ydata, sigmas="default"):
    if sigmas=="default":
        sigmas = np.full( xdata.shape, 1.0 )
    l_exp, l_c = linear_log_fit(xdata, ydata)
    return weighted_power_law_fit_const(xdata, ydata, sigmas, p0=(l_exp, l_c, 0))

def autoseeded_weighted_power_law_fit(xdata, ydata, sigmas="default"):
    if sigmas=="default":
        sigmas = np.full( xdata.shape, 1.0 )
    l_exp, l_c = linear_log_fit(xdata, ydata)
    return weighted_power_law_fit(xdata, ydata, sigmas, p0=(l_exp, l_c))

########### DFA 


def segment_fluct2(x, y, z, approx="quadratic", estimator="rms"):
    npoints = x.size

    trend_z = np.zeros((1,))
    if approx=="quadratic":        
        design = np.column_stack((x*x, y*y, x*y, x, y, np.full(npoints, 1.0)))

        param = np.array(np.linalg.lstsq(design, z)[0])
        trend_z = design @ param

    elif approx=="linear":
        design = np.column_stack((x, y, np.full(npoints, 1.0)))

        param = np.array(np.linalg.lstsq(design, z)[0])
        trend_z = design @ param

    z_residue = z - trend_z
    res = 0
    if estimator=="rms": 
        res = np.mean(z_residue**2)
    elif estimator=="range":
        res = (np.max(z_residue) - np.min(z_residue) )**2

    # if res < np.power(10.0, -15): res = np.power(10.0, -14)
    return res

def dfa_1(z2d, approx="quadratic", estimator="rms", s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False):
    (m, n) = z2d.shape
    x2d,y2d = np.mgrid[:m,:n]

    if s_max == "auto":
        s_max = n//4
    else:
        s_max = s_max


    # scales - flucts table
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
                x_segment = (x2d[s*v:s*(v+1), s*w:s*(w+1)])
                y_segment = (y2d[s*v:s*(v+1), s*w:s*(w+1)])
                z_segment = (z2d[s*v:s*(v+1), s*w:s*(w+1)])

                if np.count_nonzero(z_segment) > (s**2 * min_nonzero):

                    x_segment_2 = x_segment[ z_segment != 0.0]
                    y_segment_2 = y_segment[ z_segment != 0.0]
                    z_segment_2 = z_segment[ z_segment != 0.0]
                    f = segment_fluct2(x_segment_2.ravel(), y_segment_2.ravel(), z_segment_2.ravel(), approx=approx, estimator=estimator)
                    segment_fs[index] = f
                    index += 1
                else:
                    miss_count += 1

        # remove unfilled slots from tail
        segment_fs = np.trim_zeros(segment_fs, trim="b")
        if segment_fs.size > 4:
            fluct_av_sq = np.sum(segment_fs)/(index)
            scales_flucts[j,0] = s
            scales_flucts[j,1] = np.sqrt(fluct_av_sq)
            if messages == True:
                print( "fluct_av_sq", (fluct_av_sq), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty")
        else:
            scales_flucts[j,0] = s
            scales_flucts[j,1] = np.nan
            if messages == True:
                print( "fluct_av_sq", (np.nan), "scale", s, ",", n_submat_m*n_submat_n, "submatrices of which", miss_count, "empty. fit too poor, aborted")
    # remove nan values (from linear regression on too few data points)
    scales_flucts = scales_flucts[~np.isnan(scales_flucts).any(axis=1)]


    return scales_flucts[:,0], scales_flucts[:,1]



def dfa_H(z2d, estimator="rms", approx="quadratic", s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False, data=True):

    scales, flucts = dfa_1(z2d, estimator=estimator, approx=approx, s_min=s_min, s_max=s_max, min_nonzero=min_nonzero, messages=messages)

    # s_f_log = np.log10(scales_flucts)
    s_log = np.log10(scales)
    f_log = np.log10(flucts)

    H, c, pcov = autoseeded_weighted_power_law_fit(scales, flucts, sigmas=flucts)
    if data:
        return H, c, scales, flucts
    else:
        return H
###################### new mod ######################



def range_1(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False):
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
    return scales_flucts[:,0], scales_flucts[:,1]




def rms_1(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False):
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

    return scales_flucts[:,0], scales_flucts[:,1]


def hann_function(r, L):
    return (3*np.pi/8 - 2/np.pi)**(-0.5) *(1+np.cos((2*np.pi*r)/L) )

def fourier_1(z2d, s_min = 6, s_max = "auto", windowing=True, fill_with_mean=True ,images=False, corr=False):
    side = z2d.shape[0]

    if fill_with_mean == True:
        mean = np.mean(z2d[ z2d != 0.0 ])
        z2d[ z2d==0.0 ] = mean
    

    sx, sy = z2d.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    rad = np.hypot( X - sx/2 , Y - sy/2 )
    hann = np.zeros(rad.shape)
    h_cut = min(sx,sy)
    hann[ rad < h_cut/2 ] = hann_function( rad[ rad < h_cut/2 ], h_cut )

    if windowing == False:
        image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d))   ) **2
    else:
        image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d*hann))   ) **2

    
    if images:
        plt.figure()
        plt.imshow(z2d, interpolation="None")
        plt.show()
        plt.figure()
        plt.imshow(np.log(image_fft), interpolation="None")
        plt.show()


    if corr:
        corr2d = fftconvolve(z2d, z2d[::-1,::-1], mode="full")
        plt.figure()
        plt.imshow(corr2d, interpolation="None")
        plt.show()


    freq_points = side//2
    freqs = np.arange( 0,freq_points )

    powers = scipy_ndimage_mean(image_fft, np.round(rad), index=freqs)

    # eliminate zero-frequency component
    return freqs[1:], powers[1:]


def freq_exp_to_H(freq_exp):
    return -(freq_exp + 2.0)/2


def fourier_H(z2d, s_min = 6, s_max = "auto", cutoff ="default", windowing=True, fill_with_mean=False ,images=False, corr=False, data=True):
    freqs, powers = fourier_1(z2d=z2d, s_min=s_min, windowing=windowing, s_max=s_max, fill_with_mean=fill_with_mean ,images=images, corr=corr)

    if cutoff == "default":
        cutoff = int(freqs.size*0.8)
    freq_exp, c, pcov = autoseeded_weighted_power_law_fit(freqs[:-cutoff], powers[:-cutoff], sigmas=powers[:-cutoff])

    detect_H = freq_exp_to_H(freq_exp)
    if data:
        return detect_H, freq_exp, c, freqs, powers
    else:
        return detect_H



def higuchi_area(z2d_in, lattice_l=5):
 
    z2d = np.copy(z2d_in)

    n, m = z2d.shape

    z2d[z2d == 0.0] = np.nan

    quad_bl_z = z2d[0:n-1, 0:m-1]
    quad_br_z = z2d[0:n-1, 1:m]
    quad_tl_z = z2d[1:n, 0:m-1]
    quad_tr_z = z2d[1:n, 1:m]


    tri1_area = np.abs(quad_bl_z - quad_br_z) * np.abs( quad_bl_z - quad_tl_z)
    tri1_area = tri1_area[~np.isnan(tri1_area)]

    tri2_area = np.abs(quad_tr_z - quad_br_z) * np.abs(quad_tr_z - quad_tl_z)
    tri2_area = tri1_area[~np.isnan(tri1_area)]

    area = (np.sum(tri1_area) + np.sum(tri2_area))
    return area


def higuchi_1(z2d):

    xwid, ywid = z2d.shape
    space = min(z2d.shape)
    kmax = space//15

    scales = np.zeros(kmax-1)
    areas = np.zeros(kmax-1)

    for i, k in enumerate(np.arange(1,kmax)):
        nshifts = max(k-1,1)
        for shift_x in np.arange(0,nshifts):
            for shift_y in np.arange(0,nshifts):
                submat = z2d[ shift_x::k, shift_y::k ]
                tr_area = higuchi_area(submat, lattice_l=5*k)

                # edge correction
                n_quads = ((submat.shape[0]-1) * (submat.shape[1]-1))
                full_nquads = (xwid-1) * (ywid-1) *(1/k**2)

                areas[i] += (tr_area) *(full_nquads/n_quads) /(nshifts)**2

        scales[i] = k**2

    return scales, areas


def higuchi_exp_to_H(higuchi_exp):
    return (2 + higuchi_exp - 1)

def higuchi_H(z2d, data=True):
    scales, areas = higuchi_1(z2d)
    h_exp, h_c, pcov = autoseeded_weighted_power_law_fit(scales, areas, sigmas=areas)

    det_H = higuchi_exp_to_H(h_exp)

    if data:
        return det_H, h_exp, h_c, scales, areas
    else:
        return det_H






def range_H(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False):
    scales, flucts = range_1(z2d, s_min, s_max=s_max, min_nonzero=min_nonzero, messages=messages)

    H, c, const, pcov = autoseeded_weighted_power_law_fit(scales, flucts, sigmas=flucts)

    return H, c, const, scales, flucts

def rms_H(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.99, messages = False):
    scales, flucts = rms_1(z2d, s_min, s_max=s_max, min_nonzero=min_nonzero, messages=messages)

    H, c, const, pcov = autoseeded_weighted_power_law_fit(scales, flucts, sigmas=flucts)

    return H, c, const, scales, flucts



####### ctypes dma ########
# the .so file is in random place
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double

# input type for the cos_doubles function
# must be a double array, with single dimension that is contiguous
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')

# load the library, using numpy mechanisms
libdma = npct.load_library("libdma", "/home/gaboloth/D/fisica/tesi/comp/dma/c_dma")

# setup the return types and argument types
libdma.dma.restype = None
libdma.dma.argtypes = [array_1d_double, c_int, c_int, array_1d_double, c_int, c_int, c_double]


def dma_1(z2d, min_nonzero=0.99, s_min=5, s_max="auto"):

    M, N = z2d.shape

    if s_max == "auto":
        s_max = min((M,N))//5
    else:
        # s_max = s_max
        pass

    if s_max < s_min:
        print("too few data")
        return "too few data"

    # s_min = 5
    # s_max = 21
    s_out = np.arange(start=s_min, stop=s_max).astype(float)
    f_out = np.empty_like(s_out)

    libdma.dma(z2d.ravel(), M, N, f_out, s_min, s_max, min_nonzero)

    return s_out, f_out

def dma_H(z2d, min_nonzero=0.99, s_min=5, s_max="auto"):
    scales, flucts = dma_1(z2d, min_nonzero, s_min, s_max)

    detect_H, c, pcov = autoseeded_weighted_power_law_fit(scales, flucts, sigmas=flucts)

    return detect_H, c, scales, flucts


#### box counting

def box_counting_1(z2d, z_scale = 100000, log_spacing=False):

    xwid, ywid = z2d.shape
    space = min(z2d.shape)
    smax = space//6
    smin = 2

    # if log_spacing = False:
    scales = np.arange(smin,smax)
    nbox_s = np.zeros(smax - smin)
    if log_spacing:
        pow2N = int(np.log2(xwid)) -1
        scales = (np.logspace(0,pow2N, base=2, num=pow2N+1)).astype(int)
        nbox_s = np.zeros(scales.shape)

    s_dim = 2

    for i, s in enumerate(scales):
        n_boxes = 0
        area = s**(s_dim)
        for i_x in range(xwid//s):
            for i_y in range(ywid//s):
                submat = z2d[ i_x*s:(i_x+1)*s +1 , i_y*s:(i_y+1)*s +1 ]*z_scale
                # if (submat[ submat != 0].size)/(submat.size) > 0.90:
                n_boxes += (1 + (np.max(submat) - np.min(submat))//s )
                ######### else: zero boxes
        nbox_s[i] = n_boxes*area
        # scales[i] = s

    return scales, nbox_s


def box_counting_H(z2d, data=True, z_scale = 100000, log_spacing=False):
    scales, nbox_s = box_counting_1(z2d, z_scale=z_scale, log_spacing=log_spacing)

    num = 1
    b_exp, b_c, b_k, pcov = autoseeded_weighted_power_law_fit_const(scales[num:], nbox_s[num:])

    det_H = box_exp_to_H(b_exp)
    # fract_dim = 3 - det_H


    if data:
        return det_H, b_exp, b_c, b_k, scales, nbox_s
    else:
        return det_H

def box_exp_to_H(b_exp):
    return 1 + b_exp


# convenience function
# usage: results = logplot(fourier_H(data))
# is equivalent to just results = fourier_H(data) but 
# also shows a plot

def logplot(args):
    if len(args) == 4:
        (exp, c, x_arr, y_arr) = args
        plt.scatter(x_arr, y_arr)
        plt.plot(x_arr, power_law(x_arr, exp, c), color="red", label="H ="+str(exp))
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()
        return args
    elif len(args) == 5:
        (H, exp, c, x_arr, y_arr) = args
        plt.scatter(x_arr, y_arr)
        plt.plot(x_arr, power_law(x_arr, exp, c), color="red", label="H ="+str(H))
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()
        return args
    elif len(args) == 6:
        (H, exp, c, kon, x_arr, y_arr) = args
        plt.scatter(x_arr, y_arr)
        plt.plot(x_arr, power_law_const(x_arr, exp, c, kon), color="red", label="H ="+str(H))
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.show()
        return args
    else:
        raise("logplot: arguments not understood") 