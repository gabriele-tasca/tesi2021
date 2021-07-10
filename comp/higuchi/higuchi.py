import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import fract

def higuchi_triangulation_area(z2d_in, lattice_l=5, plots=False):
 
    z2d = np.copy(z2d_in)

    n, m = z2d.shape

    z2d[z2d == 0.0] = np.nan

    quad_bl_x, quad_bl_y = (np.mgrid[0:n-1, 0:m-1]*lattice_l).astype(np.float64)
    quad_br_x, quad_br_y = (np.mgrid[0:n-1, 1:m]*lattice_l).astype(np.float64)
    quad_tl_x, quad_tl_y = (np.mgrid[1:n, 0:m-1]*lattice_l).astype(np.float64)
    quad_tr_x, quad_tr_y = (np.mgrid[1:n, 1:m]*lattice_l).astype(np.float64)
    
    quad_bl_z = z2d[0:n-1, 0:m-1]
    quad_br_z = z2d[0:n-1, 1:m]
    quad_tl_z = z2d[1:n, 0:m-1]
    quad_tr_z = z2d[1:n, 1:m]

    quad_bl_vec = np.array( (quad_bl_x, quad_bl_y, quad_bl_z)) 
    quad_br_vec = np.array( (quad_br_x, quad_br_y, quad_br_z))
    quad_tl_vec = np.array( (quad_tl_x, quad_tl_y, quad_tl_z))
    quad_tr_vec = np.array( (quad_tr_x, quad_tr_y, quad_tr_z))

    cross1 = np.cross( quad_bl_vec - quad_br_vec, quad_bl_vec - quad_tl_vec, axisa=0, axisb=0)
    tri1_area = np.linalg.norm(cross1, axis=2)
    tri1_area = tri1_area[~np.isnan(tri1_area)]

    cross2 = np.cross( quad_tr_vec - quad_br_vec, quad_tr_vec - quad_tl_vec, axisa=0, axisb=0)
    tri2_area = np.linalg.norm(cross2, axis=2)
    tri2_area = tri2_area[~np.isnan(tri2_area)]

    area = (np.sum(tri1_area) + np.sum(tri2_area))/2.0
    return area



def area_2(z2d_in, lattice_l=5):
 
    z2d = np.copy(z2d_in)

    n, m = z2d.shape

    # z2d[z2d == 0.0] = np.nan

    quad_bl_z = z2d[0:n-1, 0:m-1]
    quad_br_z = z2d[0:n-1, 1:m]
    quad_tl_z = z2d[1:n, 0:m-1]
    quad_tr_z = z2d[1:n, 1:m]


    tri1_area =  np.sqrt( (quad_bl_z - quad_br_z)**2 + ( quad_bl_z - quad_tl_z)**2 + lattice_l**2 )
    # tri1_area = tri1_area[~np.isnan(tri1_area)]
    # print(tri1_area[0,0])

    tri2_area =  np.sqrt( (quad_tr_z - quad_br_z)**2 + (quad_tr_z - quad_tl_z)**2 + lattice_l**2 )
    # tri2_area = tri1_area[~np.isnan(tri1_area)]

    area = (0.5*lattice_l) *(np.sum(tri1_area) + np.sum(tri2_area))
    return area




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


df = pd.read_csv("/home/gaboloth/D/fisica/tesi/stats/stats.csv", sep=";")
df = pd.read_csv("stats.csv", sep=";")

# df["Nome"]

# name = "Cima Wanda"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

# res = np.zeros(9)
# stat_n = 50
# for i,h in enumerate(np.linspace(0.1, 0.9, 9)):
#     for j in range(stat_n):
#         z2d = fract.fbm2D(H=h, N=8)
#         res[i] += (area2(z2d))/stat_n
# plt.plot(res)


N = 10
# # z2d = np.full((2**N +1, 2**N +1), 35.0)
z2d = fract.fbm2D(H=0.6, N=N)
z2d = fract.cut_profile(z2d, contour_H=0.99)

# name = "Scerscen Inferiore"
# z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

plt.figure()
plt.imshow(z2d)
plt.show()
plt.figure()

def higuchi_H(z2d):
    pass

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


h_exp, h_c, pcov = fract.autoseeded_weighted_power_law_fit(scales[30:], areas[30:], sigmas=areas[30:])

det_H = 2 - (-h_exp +1 ) 

    # return det_H, h_exp, h_c, scales, areas


# det_H, h_exp, h_c, scales, areas = higuchi_H(z2d)

print("fractal dimension:", -h_exp+1)
plt.scatter(scales, areas)
plt.plot(scales, fract.power_law(scales, h_exp, h_c), color="red")
plt.xscale("log")
plt.yscale("log")
plt.title("Higuchi: H = "+str(det_H))
plt.show()





# def to_data(name, inner_func, *args, **kwargs):
#     npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
#     return inner_func(npy_points, *args, **kwargs)

# def higuchi_H_discard(*args, **kwargs):
#     try:
#         det_H, _h_c, _scales, _areas = higuchi_H(*args, **kwargs)
#         print(det_H)
#         return det_H
#     except Exception as ex:
#         print(ex)
#         return np.nan

# df["Higuchi H (paper)"] = df["Nome"].apply(to_data, inner_func=higuchi_H_discard)


# a = df[ df["Area"] < 97513  ]
# plt.scatter(a["Fourier H"], a["Higuchi H (paper)"])

# plt.scatter(df["DFA H"], df["Higuchi H (paper)"])

# df.to_csv("stats.csv", sep=";", index=False)