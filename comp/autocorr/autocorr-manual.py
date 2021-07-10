
import numpy as np
import matplotlib.pyplot as plt

import fract
import scipy.signal

import pandas as pd


df = pd.read_csv("/home/gaboloth/D/fisica/tesi/stats/2.0/stats.csv", sep=";")

df["Nome"]

H = 0.6
N = 9

npoints = 100000
magic_number = 0.47


# z2d = fract.fbm2D_exact(H, N)


# flat plane
X, Y = np.mgrid[0:300, 0:300]
z2d = np.full(X.shape, 10.0) + Y.astype(float)*0.001
# z2d = Y.astype(float) + 0.5*X.astype(float)
plt.figure()
plt.imshow(z2d)
plt.show()




name = "Forni"
print(name)
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")


# # generate exp-corr noise
# half_beta = - (H + 1)
# n = 2**N
# exp_r = 10.0
# rng = np.random.default_rng()
# synth_spectrum_quarter = rng.standard_normal((n//2,n//2)) + 1j * rng.standard_normal((n//2,n//2))
# tx = np.arange(0,n//2)
# ty = np.arange(0,n)
# radius = np.sqrt( ( tx.reshape((1,n//2))  )**2 + (ty.reshape((n,1)) - n/2 )**2  )
# radius = np.vstack(( radius[n//2:n, :], radius[0:n//2 , :] ))
# synth_spectrum = np.vstack( (synth_spectrum_quarter, synth_spectrum_quarter[::-1]) )
# # coeff = 4*np.pi* (tx)**(half_beta+2) /(half_beta+2)
# coeff = 10**(N/2)
# bell =  np.exp(-(radius + 0.01)/exp_r) * coeff
# filtered_synth_spectrum = synth_spectrum * bell
# z2d = (np.fft.irfft2(filtered_synth_spectrum))
# z2d = z2d - np.mean(z2d)





## subtract mean plane
# X, Y = np.mgrid[0:xwid, 0:ywid]
# design = np.column_stack((X.ravel(), Y.ravel(), np.full(xwid*ywid, 1.0)))
# param = np.array(np.linalg.lstsq(design, z2d.ravel())[0])
# trend_z = design @ param
# trend_z = trend_z.reshape(z2d.shape)
# z2d[z2d!=0.0] = z2d[z2d!=0.0] - trend_z[z2d!=0.0]

# plt.figure()
# plt.imshow(z2d)
# plt.show()


### fft corr
xwid, ywid = z2d.shape

fft_corr2d = scipy.signal.fftconvolve(z2d, z2d[::-1,::-1], mode="full")

sx, sy = fft_corr2d.shape
max_r = min(sx,sy)/4
X, Y = np.ogrid[0:sx, 0:sy]
rad = np.hypot( X - sx/2 , Y - sy/2 )
fft_r_points = np.arange(1,max_r)
fft_autocorrs = scipy.ndimage.mean(fft_corr2d, np.round(rad), index=fft_r_points)

plt.figure()
plt.plot(fft_r_points, (fft_autocorrs))
plt.title("fft")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"C(r)")
plt.xlabel(r"r")
plt.title("FFT autocorrelation")
plt.show()
# plt.savefig("forni-linear-FFT-autocorr.png",  bbox_inches='tight')






# def manual_correlation(z2d):


### subtract mean 
mean = np.mean(z2d[ z2d != 0.0 ])
z2d[z2d!=0.0] -= mean


max_r = min(z2d.shape)/3

r_points = np.arange(0.0,int(max_r))
autocorrs = np.zeros(int(max_r))

for r in range(0,int(max_r)):
    misspercent = (r/max_r)*magic_number
    # loc_npoints = npoints 
    loc_npoints = int(npoints * (1 + misspercent))
    # print(loc_npoints)
    x1 = np.random.randint(0,xwid, size=loc_npoints)
    y1 = np.random.randint(0,ywid, size=loc_npoints)
    thetas = np.random.uniform(0,2*np.pi, size=loc_npoints)
    x2 = (x1 + np.cos(thetas)*r).astype(np.int)
    y2 = (y1 + np.sin(thetas)*r).astype(np.int)
    
    mask = np.array((x2 > 0) & (x2 < xwid) & (y2 > 0) & (y2 < ywid) )
    x1 = x1[mask]
    y1 = y1[mask]
    x2 = x2[mask]
    y2 = y2[mask]

    value1 = z2d[x1,y1]
    value2 = z2d[x2,y2]
    mask2 = np.array( (value1 != 0.0) & (value2 != 0.0) )
    value1 = value1[mask2]
    value2 = value2[mask2]
    prod = value1 * value2
    autocorrs[r] = np.mean(prod)
    



# exp, c, pcov = fract.autoseeded_weighted_power_law_fit(r_points[1:20], autocorrs[1:20], sigmas=autocorrs[1:20])
# print(exp)
plt.figure()
plt.plot(r_points, autocorrs)
# plt.plot(r_points, fract.power_law(r_points, exp, c), color="orange")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"C(r)")
plt.xlabel(r"r")
plt.title("space-domain autocorrelation")
plt.show()
# plt.savefig("forni-linear-autocorr.png",  bbox_inches='tight')


