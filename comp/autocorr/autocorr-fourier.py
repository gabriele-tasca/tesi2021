
import numpy as np
import matplotlib.pyplot as plt

import fract
import scipy.signal

H = 0.8
N = 9


z2d = fract.fbm2D_midpoint(H, N)
# z2d = fract.cut_profile(z2d, 0.99)

max_r = min(z2d.shape)/2

# # fill_with_mean 
# mean = np.mean(z2d[ z2d != 0.0 ])
# z2d[ z2d==0.0 ] = mean
# subtract mean 
mean = np.mean(z2d[ z2d != 0.0 ])
z2d[ z2d!=0.0 ] -= mean


def hann_function(r, L):
    return (3*np.pi/8 - 2/np.pi)**(-0.5) *(1+np.cos((2*np.pi*r)/L) )


sx, sy = z2d.shape
X, Y = np.ogrid[0:sx, 0:sy]

rad = np.hypot( X - sx/2 , Y - sy/2 )
hann = np.zeros(rad.shape)
h_cut = min(sx,sy)
hann[ rad < h_cut/2 ] = hann_function( rad[ rad < h_cut/2 ], h_cut )

# windowing?
image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d*hann))   ) **2
# image_fft = abs( np.fft.fftshift(np.fft.fft2(z2d))   ) **2

corr2d = scipy.signal.fftconvolve(z2d, z2d[::-1,::-1], mode="full")

# plt.figure()
# plt.imshow(z2d, interpolation="None")
# plt.show()
plt.figure()
plt.imshow(corr2d, interpolation="None")
plt.show()

r_points = np.arange(1,max_r)

autocorrs = scipy.ndimage.mean(image_fft, np.round(rad), index=r_points)

exp, c, pcov = fract.autoseeded_weighted_power_law_fit(r_points, autocorrs, sigmas=autocorrs)
plt.scatter(r_points, autocorrs)
plt.plot(r_points, fract.power_law(r_points, exp, c), color="orange")
plt.xscale("log")
plt.yscale("log")
plt.show()

det_H =  -exp/2 -1
print("gen_H", H)
print("det_H", det_H)



