import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def triangulation_area(z2d_in, lattice_l=5, flat=False, map_flat=False, plots=False):
 
    z2d = np.copy(z2d_in)

    n, m = z2d.shape

    z2d[z2d == 0.0] = np.nan

    if flat:
 


        # replace all points with the mean plane
        x2d, y2d = (np.mgrid[0:n, 0:m])


        x = x2d[~np.isnan(z2d)].ravel()
        y = y2d[~np.isnan(z2d)].ravel()
        z = z2d[~np.isnan(z2d)].ravel()




        design = np.column_stack((x, y, np.full(x.size, 1.0)))

        param = np.array(np.linalg.lstsq(design, z)[0])
        trend_z = design @ param

        z_residue = z - trend_z


        if plots:
            trend_z2d = np.copy(z2d)
            for i, el in enumerate(z_residue):
                i_x = x[i]
                i_y = y[i]
                trend_z2d[i_x, i_y] = trend_z[i]
            #####################
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.plot_surface(x2d,y2d, trend_z2d, alpha=0.5)
            ax.scatter(x, y, z, marker=".")
            plt.show()
            #####################


        for i, el in enumerate(z_residue):
            i_x = x[i]
            i_y = y[i]
            z2d[i_x, i_y] = trend_z[i]


        if plots:
            #####################
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.scatter(x, y, trend_z)
            ax.plot_surface(x2d,y2d, trend_z2d, alpha=0.5)
            plt.show()
            #####################
    elif map_flat:
        z2d[ np.isnan(z2d) == False ] = 15.0




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


def triangulation_area_name(name, *args, **kwargs):
    npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")
    return triangulation_area(npy_points, *args, **kwargs)




def z_stdev(z2d):
    return np.std( z2d[ z2d != 0.0 ] )
# df["Height Stdev"] = df["Nome"].apply(to_data, inner_func=z_stdev)



df = pd.read_csv("stats.csv", sep=";")

# df["Nome"]

# name = "Cima Wanda"
# npy_points = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

# flat = triangulation_area(npy_points, flat=True, plots=True)

# real = triangulation_area(npy_points, flat=False, plots=True)
# print("real",real) 
# print("flat",flat)

# real/flat



df["Triangulation Area"] = df["Nome"].apply(triangulation_area_name)

df["Flat Triangulation Area"] = df["Nome"].apply(triangulation_area_name, flat=True)

df["Map Area"] = df["Nome"].apply(triangulation_area_name, map_flat=True)

df["Area Ratio"] = df["Triangulation Area"]/df["Flat Triangulation Area"]

df["Map Area Ratio"] = df["Triangulation Area"]/df["Map Area"]

df.to_csv("stats.csv", sep=";", index=False)