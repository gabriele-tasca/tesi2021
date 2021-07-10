import numpy as np
import matplotlib.pyplot as plt

import scipy.signal

name = "Narcanello"
print(name)
z2d = np.load("/home/gaboloth/D/fisica/tesi/dati/npysquare/all/"+name+"-2d.npy")

z2d[z2d != 0.0] -= np.mean(z2d[z2d != 0.0] )

xwid, ywid = z2d.shape

deltax = 50
deltay = 0

displayz = np.zeros((xwid+deltax,ywid+deltay+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++))

leftz = z2d[:, deltax:]
rightz = z2d[:, :(ywid-deltax) ]

prod = np.multiply(leftz,rightz)


plt.figure()
plt.imshow(z2d)
plt.show()
plt.figure()
plt.imshow(prod)
plt.show()
