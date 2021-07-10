import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

import fract
N = 10
H = 0.7

def fbm2D_spectral(H, N):
    half_beta = - (H + 1)

    n = 2**N

    rng = default_rng()
    synth_spectrum_quarter = rng.standard_normal((n//2,n//2)) + 1j * rng.standard_normal((n//2,n//2))

    tx = np.arange(0,n//2)
    ty = np.arange(0,n)

    radius = np.sqrt( ( tx.reshape((1,n//2))  )**2 + (ty.reshape((n,1)) - n/2 )**2  )
    radius = np.vstack(( radius[n//2:n, :], radius[0:n//2 , :] ))

    synth_spectrum = np.vstack( (synth_spectrum_quarter, synth_spectrum_quarter[::-1]) )

    # coeff = 4*np.pi* (tx)**(half_beta+2) /(half_beta+2)
    coeff = 10**(N/2 +1)
    bell =  np.power(radius + 0.01, half_beta) * coeff

    filtered_synth_spectrum = synth_spectrum * bell

    synth_image = (np.fft.irfft2(filtered_synth_spectrum))

    return synth_image

# gen_H_s = np.arange(0.1,0.9,0.1)
gen_H_s = np.array([0.7])
det_H_s = np.zeros(gen_H_s.shape)

for i,H in enumerate(gen_H_s):
    z2d = fbm2D_spectral(H,N+1)
    # rsh = 241 # random shift
    # cut = int(2**N*0.743)
    # z2d = z2d_big[rsh:cut+rsh,rsh:cut+rsh]

    plt.figure()
    plt.imshow(z2d)
    plt.title("synth image")
    plt.show()

    detect_H, freq_exp, c, freqs, powers  = fract.logplot(fract.fourier_H(z2d, windowing=False))
    det_H_s[i] = detect_H
    print("gen H", H)
    print("fourier detect_H ", detect_H)

    print("max", np.max(z2d))
    print("min", np.min(z2d))
    print("range", np.abs(np.abs(np.max(z2d))-np.abs(np.min(z2d))))


plt.plot(gen_H_s,gen_H_s)
plt.scatter(gen_H_s,det_H_s)


# midpoint_image = fract.fbm2D_midpoint(H, N)
# plt.figure()
# plt.imshow((midpoint_image))
# plt.title("midpoint image")
# plt.show()

# exact_image = fract.fbm2D_exact(H, 2**N)
# plt.figure()
# plt.imshow((exact_image))
# plt.title("exact image")
# plt.show()

#############
##### detect
#############



# dfa_detect_H, dfa_c, dfa_const, dfa_freqs, dfa_powers  = fract.dfa_H(synth_image)
# print("dfa detect_H ",dfa_detect_H) 
