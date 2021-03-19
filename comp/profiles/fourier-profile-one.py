import numpy as np
import scipy.stats
import scipy.optimize
import matplotlib.pyplot as plt

from scipy.signal import correlate2d
import fract

def pow_law(x, a, b):
    return a * np.power(x,b)
    # return a * x**(b)


def log_sample(x, y, base=2):
    ind = np.array(range(x.size))
    xl = x[ (np.mod(np.log(ind)/np.log(base), 1)) == 0 ]
    yl = y[ (np.mod(np.log(ind)/np.log(base), 1)) == 0 ]

    return(xl, yl)


if __name__ == '__main__':
    print("generating")
    N = 8
    gen_H = np.float64(0.1)
    contour_H = np.float64(0.7)
    z2d = fract.fbm2D(gen_H, N)
    z2d = fract.cut_profile(z2d, contour_H)

    imm = np.copy(z2d)
    imm[imm == 0.0] = np.nan
    plt.imshow(imm, interpolation="None")
    plt.show()

    freqs, powers = fract.fourier_1(z2d, images=False, corr=False)

    freq_exp, c = fract.linear_log_fit(freqs[1:], powers[1:])
    detect_H = fract.freq_exp_to_H(freq_exp)

    freqs = freqs[2:]
    powers = powers[2:]
    sigmas = powers

    popt, pcov = scipy.optimize.curve_fit(pow_law, freqs, powers, sigma=sigmas,  p0=(1.0,-3.0))
    print( "scipy H",  -(popt[1] + 1.5)/2 )



    expected_freq_exp = -1.5 - 2.0*gen_H
    print("freq_exp", freq_exp)
    print("expected_freq_exp", expected_freq_exp)

    plt.scatter(freqs,powers, marker=".", color="deepskyblue")
    # lin regression on log data
    plt.plot(freqs, (c)*freqs**freq_exp, color="purple")
    
    # generation H
    plt.plot(freqs, c*freqs**expected_freq_exp, color="red")

    # weighted nonlinear least squares
    plt.plot(freqs, pow_law(freqs, *popt), color="springgreen")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("H = " + str(detect_H))
    plt.show()

    plt.scatter(freqs,powers, marker=".", color="deepskyblue")
    plt.plot(freqs, (c)*freqs**freq_exp, color="purple")
    plt.plot(freqs, c*freqs**expected_freq_exp, color="red")
    plt.plot(freqs, pow_law(freqs, *popt), color="springgreen")


    plt.yscale("log")
    plt.xscale("linear")
    plt.title("H = " + str(detect_H))
    plt.show()

