
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
    N = 9
    gen_H = np.float64(0.7)
    contour_H = np.float64(0.7)
    z2d = fract.fbm2D(gen_H, N)
    z2d = fract.cut_profile(z2d, contour_H)

    imm = np.copy(z2d)
    imm[imm == 0.0] = np.nan
    plt.imshow(imm, interpolation="None")
    plt.show()

    scales, flucts = fract.rms_1(z2d)

    det_H, c, = fract.linear_log_fit(scales, flucts)

    # popt, pcov = scipy.optimize.curve_fit(pow_law, scales, flucts, sigma=sigmas,  p0=(1.0,-3.0))
    sci_H, sci_C, pcov = fract.weighted_power_law_fit(scales,flucts, sigmas= flucts, p0=(1,0.5))
    print( "scipy det_H",  sci_H )



    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (c)*scales**det_H, color="purple")
    plt.plot(scales, c*scales**gen_H, color="red")
    plt.plot(scales, sci_C*scales**sci_H, color="springgreen")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("det_H = " + str(det_H))
    plt.show()

    plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    plt.plot(scales, (c)*scales**det_H, color="purple")
    plt.plot(scales, c*scales**gen_H, color="red")
    plt.plot(scales, sci_C*scales**sci_H, color="springgreen")


    plt.yscale("log")
    plt.xscale("linear")
    plt.title("det_H = " + str(det_H))
    plt.show()


    ############# end #############

    # print("det_H =  ",det_H)

    # det_H, c, scales, flucts = profile_fourier_from_z2d(z2d, messages=True)

    # plt.scatter(scales,flucts, marker=".", color="deepskyblue")
    # plt.plot(scales, (10**c)*scales**det_H, color="purple")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("det_H = " + str(det_H))
    # plt.show()