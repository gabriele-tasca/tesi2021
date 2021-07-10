import numpy as np
import matplotlib.pyplot as plt

# import fourier
import fract


def cut_profile(image, contour_H, treshold=0.0, invert=1):
    n, m = image.shape
    side = max(n,m)
    N = np.ceil(np.log2(side-1)).astype(int)
    contour_im = fract.fbm2D(contour_H, N=N)
    contour_im[ invert*contour_im >= invert*treshold ] = 0.0
    contour_im[ invert*contour_im < invert*treshold ] = 1.0
    return np.multiply(contour_im, image)


# image = fract.fbm2D(0.6, 8)
# plt.imshow(image)
# plt.show()
# image = cut_profile(image, 0.7)
# plt.imshow(image)
# plt.show()


#######

N = 9

stat_n = 100

start = 0.1
stop = 0.9001
step = 0.1
nhs = len(np.arange(start,stop=stop,step=step))
res = np.zeros( (nhs, 3))

for data_H in np.arange(start,stop=stop,step=step):

    for (i, contour_H) in enumerate( np.arange(start,stop=stop,step=step) ):

        stat_res = np.zeros(stat_n)
        # print("contour_H", contour_H)
        for j in range(1, stat_n):

            # generate
            # Random.seed!(729 + i*34);
            data = fract.fbm2D(data_H,N=N)
            data = cut_profile(data, contour_H)


            # dfa
            h_detected, freq_exp, c, freqs, powers = fract.profile_fourier_from_z2d(data)
            stat_res[j] = h_detected
            # print("     ", j)


        # just for show
        # sample_data = fract.fbm2D(data_H,N=N)
        # sample_data = cut_profile(sample_data, contour_H)
        # sample_data[ sample_data == 0.0] = np.nan
        # plt.imshow(sample_data)
        # plt.show()

        av_h_detected = np.mean(stat_res)
        std_h_detected = np.std(stat_res)
        # print("     av. H detected =", av_h_detected)
        
        newrow = np.transpose([contour_H, av_h_detected, std_h_detected])
        res[i,:] = newrow

    # print(res)
    plt.errorbar(res[:,0], res[:,1], yerr=res[:,2], color="deepskyblue", label="detected H");
    plt.xlabel("contour H")
    plt.ylabel("detected H")
    linex = np.array([0.075,0.925])
    contour_liney = np.array([0.075,0.925])
    data_liney = np.array([data_H,data_H])
    plt.plot(linex, data_liney, color="orange", label="data H")
    plt.plot(linex, contour_liney, color="springgreen", label="contour H")
    # plt.plot(linex, (contour_liney+data_liney)/2 - 0.26, color="red", label="fit")
    plt.legend()
    plt.title("data H: "+ str(data_H))
    plt.show()

# res10 = res

# past_reses = [res8, res9, res10]

# for i,r in enumerate(past_reses):
#     plt.errorbar(r[:,0], r[:,1], yerr=r[:,2], label=str(8+i));
#     plt.plot(linex, liney)
# plt.xlabel("generation H")
# plt.ylabel("detected H")
# linex = [0.075,0.925]
# liney = [0.075,0.925]


# a = np.array([1,np.nan,3])
# b = np.array([4,5,6])

# c = np.column_stack((a,b))
# c
# c.shape

# c = c[~np.isnan(c).any(axis=1)]
# c

# np.log10(c)[1]
print("over")