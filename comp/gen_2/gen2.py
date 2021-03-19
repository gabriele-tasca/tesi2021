import numpy as np
import matplotlib.pyplot as plt

import fract


def fbm2D_range(H, N=4, base_stdev=1.0, min_scale=0, max_scale=None):
    side = 2**N + 1
    rng = np.random.default_rng()

    if max_scale == None:
        max_scale = side

    vec = np.zeros( (side,side) )
    for g in range(1,N+1):
        gpow = 2**g
        gstep = (side-1) // gpow
        stdev = (0.5)**(H*g) * base_stdev

        # random additions
        if gpow > min_scale and gpow < max_scale:
            rands = rng.normal( size=(gpow+1,gpow+1), scale=stdev)
            vec[ 0:side:gstep, 0:side:gstep ] += rands

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


            #   bulk
            up3 = vec[  0:side-half:gstep, gstep:side-gstep:gstep  ]
            down3 = vec[  2*half:side+half:gstep, gstep:side-gstep:gstep  ]
            left3 = vec[ half:side:gstep, half:side-half-gstep:gstep]
            right3 = vec[ half:side:gstep, half+gstep:side-half:gstep]
            vec[  half:side:gstep, gstep:side-gstep:gstep  ] = (up3 + down3 + left3 + right3)/4
            
            ##### interpolation 2.2


            # bulk
            left4 = vec[  gstep:side-gstep:gstep, 0:side-half:gstep ]
            right4 = vec[  gstep:side-gstep:gstep, half+half:side+half:gstep  ]
            up4 = vec[ half:side-half-gstep:gstep, half:side:gstep]
            down4 = vec[ half+gstep:side-half:gstep, half:side:gstep]

            vec[  gstep:side-gstep:gstep, half:side:gstep  ] = (up4 + down4 + left4 + right4)/4


    return vec[1:side-1,1:side-1]

scale_t = 65
# small scale fractal
# z2d = fbm2D_range(0.6, N=9, max_scale=scale_t)

# large scale fractal
# z2d = fbm2D_range(0.6, N=9, min_scale=25)

# mixed scale
# small
z2d = fract.fbm2D(0.6, N=9)
# large
z2d += fract.fbm2D(0.1, N=9)


## fourier
fourier_H, freq_exp, f_c, freqs, powers = fract.fourier_H(np.copy(z2d), fill_with_mean=True, images=True, corr=True)
print("fourier_H", fourier_H)


#### dfa
dfa_H, c, scales3, flucts3 = fract.dfa_H(z2d, messages=False, min_nonzero=0.99)
print("dfa_H", dfa_H)


# fourier plot
freq_exp_2, fc2, pcov = fract.autoseeded_weighted_power_law_fit(freqs[:20], powers[:20], powers[:20])


plt.figure()
plt.scatter(freqs, powers)
plt.plot(freqs, fract.power_law(freqs, freq_exp, f_c), color="red", label="fourier: H ="+str(fract.freq_exp_to_H(freq_exp)))

plt.plot(freqs, fract.power_law(freqs, freq_exp_2, fc2), color="springgreen", label="fourier: H ="+str(fract.freq_exp_to_H(freq_exp_2)))

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

### branches
scale_f1 = 17
scale_f2 = 25

dfa_H2, c2, pcov = fract.autoseeded_weighted_power_law_fit(scales3[:scale_f1], flucts3[:scale_f1], flucts3[:scale_f1])
dfa_H, c, pcov = fract.autoseeded_weighted_power_law_fit(scales3[scale_f2:], flucts3[scale_f2:], flucts3[scale_f2:])
print("dfa_H2", dfa_H2)


plt.figure()
plt.scatter(scales3, flucts3)
plt.plot(scales3, fract.power_law(scales3, dfa_H, c), color="springgreen", label="dfa: H ="+str(dfa_H))
plt.plot(scales3, fract.power_law(scales3, dfa_H2, c2), color="red", label="dfa: H = "+str(dfa_H2))

plt.plot(scales3, fract.power_law(scales3, fourier_H, c2/0.8), color="purple", label="fourier: H = "+str(fourier_H))

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

