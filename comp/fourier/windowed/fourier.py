import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import correlate2d
from scipy.ndimage import mean as scipy_ndimage_mean
import fract

N = 10
cutoff = int((2**N)/2.5)
gen_H_s = np.arange(0.1,0.9,0.1)
det_H_s = np.zeros(gen_H_s.shape)

for i, gen_H in enumerate(gen_H_s):
    # print("generating")
    z2d = fract.fbm2D_exact(gen_H, N)

    # plt.figure()
    # plt.imshow(z2d, interpolation="None")
    # plt.show()


    side = z2d.shape[0]

    # corr2d = scipy.signal.fftconvolve(z2d, z2d[::-1,::-1], mode="full")
    # plt.imshow(corr2d, inter\polation="None")
    # plt.show()

    sx, sy = z2d.shape
    X, Y = np.ogrid[0:sx, 0:sy]

    def hann_function(r, L):
        return (3*np.pi/8 - 2/np.pi)**(-0.5) *(1+np.cos((2*np.pi*r)/L) )

    rad = np.hypot( X - sx/2 , Y - sy/2 )
    hann = np.zeros(rad.shape)
    h_cut = min(sx,sy)
    hann[ rad < h_cut/2 ] = hann_function( rad[ rad < h_cut/2 ], h_cut )

    # windowing
    # image_fft = np.abs( np.fft.fftshift(np.fft.fft2(z2d*hann)) ) **2.0
    # no windowing
    image_fft = np.abs( np.fft.fftshift(np.fft.fft2(z2d)) ) **2.0


    freq_points = side//2
    freqs = np.arange( 0,freq_points )

    powers = scipy_ndimage_mean(image_fft, np.round(rad), index=freqs)


    autoexp, autoc, autopcov = fract.autoseeded_weighted_power_law_fit(freqs[2:-cutoff], powers[2:-cutoff], sigmas=powers[2:-cutoff])

    tailexp, tailc, tailpcov = fract.autoseeded_weighted_power_law_fit(freqs[-cutoff:], powers[-cutoff:], sigmas=powers[-cutoff:])

    detect_H = fract.freq_exp_to_H(autoexp)
    print("expected H", gen_H)
    print("    detect_H",detect_H)
    det_H_s[i] = detect_H


    plt.figure()
    plt.scatter(freqs,powers, marker=".", color="deepskyblue")
    plt.plot(freqs, fract.power_law(freqs, autoexp, autoc), color="red")
    plt.plot(freqs, fract.power_law(freqs, tailexp, tailc), color="green")
    # plt.plot(freqs, (10**c)*freqs**expected_freq_exp, color="red")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("detect_H = %g, gen_H = %g, tail exp = %g" % (detect_H, gen_H, tailexp) )
    plt.show()


plt.figure()
plt.plot(gen_H_s, gen_H_s)
plt.scatter(gen_H_s, det_H_s)
plt.show()