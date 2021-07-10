import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt

import scipy.ndimage

from scipy.signal import correlate2d
import fract



def profile_fourier_from_z2d(z2d, s_min = 6, s_max = "auto", min_nonzero = 0.9, images=False, corr=False):
    side = z2d.shape[0]

    if corr:
        corr2d = scipy.signal.fftconvolve(z2d, z2d[::-1,::-1], mode="full")

        if images:
            plt.imshow(corr2d, interpolation="None")
            plt.show()

    # p_spectrum = abs( np.fft.fftshift(np.fft.fft2(corr2d))   ) **2
    # plt.imshow(np.log(p_spectrum), interpolation="None")
    # plt.show()

    image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d))   ) **2
    
    if images:
        plt.imshow(np.log(image_fft), interpolation="None")
        plt.show()




    values = image_fft
    # values = p_spectrum
    sx, sy = values.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    distances = np.hypot( X - sx/2 , Y - sy/2 )

    # freq_points = int(side /np.sqrt(2) )
    freq_points = side//2
    freqs = range( 0,freq_points )
    powers = np.zeros( freq_points )


    for r in range(0,freq_points):
        powers[r] = np.sum( values[  (distances > (r-0.5) ) & (distances <= r+0.5)  ] )

    ### ?????????????
    # powers = scipy.ndimage.sum(values, distances, index=np.arange(0, freq_points))

    # print(powers)

    freq_log = np.log10(freqs)
    pow_log = np.log10(powers)

    freq_exp, c, r_value, p_value, std_err = scipy.stats.linregress(freq_log[1:], pow_log[1:])

    detect_H = -(freq_exp + 2.0)/2
    print("detect_H",detect_H)
    # wrong_H = -(freq_exp + 1.5)/2
    # print("wrong_H",wrong_H)
    return detect_H, freq_exp, c, freqs, powers

def pow_law(x, a, b):
    return a * np.power(x,b)
    # return a * x**(b)

def pow_law_jacobian(x, a, b):
    return b *a * x**(b-1)  



if __name__ == '__main__':
    print("generating")
    N = 10
    gen_H = np.float64(0.8)
    z2d = fract.fbm2D_midpoint(gen_H, N)

    plt.imshow(z2d, interpolation="None")
    plt.show()


    detect_H, freq_exp, c, freqs, powers = profile_fourier_from_z2d(z2d, images=False, corr=True)

    freqs = freqs[2:]
    powers = powers[2:]
    sigmas = powers

    popt, pcov = scipy.optimize.curve_fit(pow_law, freqs, powers, sigma=sigmas, p0=(1.0,-3.0))
    # popt, pcov = scipy.optimize.curve_fit(pow_law, freqs, powers, p0=(1.0,3.0))
    popt

    
    
    # plt.plot(freqs, pow_law(freqs, *popt), color="green")


    autoexp, autoc, autopcov = fract.autoseeded_weighted_power_law_fit(freqs, powers, sigmas=sigmas)
    plt.plot(freqs, pow_law(freqs, autoexp, autoc), color="gray")


    print("expected H", gen_H)

    plt.scatter(freqs,powers, marker=".", color="deepskyblue")
    plt.plot(freqs, (10**c)*freqs**freq_exp, color="purple")
    # plt.plot(freqs, (10**c)*freqs**expected_freq_exp, color="red")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("H = " + str(detect_H))
    plt.show()

    plt.scatter(freqs,powers, marker=".", color="deepskyblue")
    plt.plot(freqs, (10**c)*freqs**freq_exp, color="purple")
    # plt.plot(freqs, (10**c)*freqs**expected_freq_exp, color="red")
    # plt.plot(freqs, pow_law(freqs, *popt), color="green")

    plt.yscale("log")
    plt.xscale("linear")
    plt.title("H = " + str(detect_H))
    plt.show()


    ############# end #############

    # print("H =  ",H)

    # H, c, scales, flucts = profile_fourier_from_z2d(z2d, messages=True)

    # plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    # plt.plot(scales, (10**c)*scales**H, color="purple")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("H = " + str(H))
    # plt.show()
