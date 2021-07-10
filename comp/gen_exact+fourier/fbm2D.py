import numpy as np
import matplotlib.pyplot as plt

import fract

# rng = np.random.default_rng()

# n_out = 512

# H = 0.76
# alpha = 2*H



# print("set params")

# beta = 0
# c0 = 0
# c2 = 0
# R = 0
# n_real = 0
# if alpha <= 1.5: 
#     R = 1
#     beta = 0
#     c2 = alpha/2
#     c0 = 1-alpha/2
#     n_real = n_out
# else: 
#     R = 2
#     beta = alpha*(2-alpha)/(3*R*(R**2-1))
#     c2 = (alpha-beta*(R-1)**2*(R+2))/2
#     c0 = beta*(R-1)**3+1-c2
#     n_real = 2*n_out


# print("long loop")

# n = (np.ceil(n_real*np.sqrt(2))).astype(int)
# m = (np.ceil(n_real*np.sqrt(2))).astype(int)
# tx = (np.arange(0,n)/n)*R
# ty = (np.arange(0,m)/m)*R


# rows = np.zeros((m,n))


# rs = np.sqrt( ( tx.reshape((n,1)) )**2 + (ty.reshape((1,m)))**2  )

# rows[ rs <= 1] = c0-rs[rs<=1]**alpha+c2*rs[rs<=1]**2
# rows[ (rs > 1 )& (rs <= R)] = beta*(R-rs[(rs>1)&(rs<=R)])**3/rs[(rs>1)&(rs<=R)]

# # ^ this might be hard to read, but it's just a np-vectorized
# # application of this function to the "rows" array:
# # def rho(p1, p2):
# #     # create continuous isotropic function
# #     r = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
# #     if r <= 1:
# #         out=c0-r**alpha+c2*r**2
# #     elif r<=R:
# #         out=beta*(R-r)**3/r
# #     else:
# #         out=0
# #     return out


# print("stacking")

# loc_a1 = rows
# loc_a2 = rows[:, -1:1:-1]
# loc_b1 = rows[-1:1:-1, :]
# loc_b2 = rows[-1:1:-1 , -1:1:-1]

# loc_c1 = np.hstack([loc_a1, loc_a2])
# loc_c2 = np.hstack([loc_b1, loc_b2])
# block_circ = np.vstack([loc_c1, loc_c2])

# print("eigenvalues")

# # compute eigen-values
# lam = np.real(np.fft.fft2(block_circ))/(4*(m-1)*(n-1))
# lam = np.sqrt(lam)
# Z1 = rng.standard_normal((2*(m-1),2*(n-1)))
# Z2 = rng.standard_normal((2*(m-1),2*(n-1)))

# print("second fft2")

# # generate field with covariance given by block circular matrix
# Z = Z1 + 1j*Z2
# F = np.fft.fft2(lam * Z)
# print("generate field ")

# F = F[0:m,0:n] # extract sub-block with desired covariance
# # (out,c0,c2) = rho( (0,0) ,(0,0),R,2*H)
# field1 = np.real(F)
# field2 = np.imag(F) # two independent fields
# field1 = field1 - field1[0,0] # set field zero at origin
# field2 = field2 - field2[0,0] # set field zero at origin

# print("corrections")

# # make correction for embedding with a term c2*r^2

# loc1 = ty.conj().T *rng.standard_normal()
# loc2 = tx*rng.standard_normal()
# loc3 = np.kron(loc1, loc2).reshape((loc1.size, loc2.size))
# field1 = field1 + loc3*np.sqrt(2*c2)
# field2 = field2 + loc3*np.sqrt(2*c2)


# if alpha <= 1.5: 
#     n_cut = int(n/(np.sqrt(2)))
#     m_cut = int(m/(np.sqrt(2)))

#     field1 = field1[ 0:n_cut, 0:m_cut ]
#     field2 = field2[ 0:n_cut, 0:m_cut ]
# else:
#     n_cut = int(n/(np.sqrt(2)*2))
#     m_cut = int(m/(np.sqrt(2)*2))

#     field1 = field1[ 0:n_cut, 0:m_cut ]
#     field2 = field2[ 0:n_cut, 0:m_cut ]



# plt.imshow(field1)
# plt.show()



# print(field1.shape )



#####################
#####################
#####################
#####################

def fbm2D_exact(H, n_out):

    rng = np.random.default_rng()

    alpha = 2*H

    #("set params")
    beta = 0
    c0 = 0
    c2 = 0
    R = 0
    n_real = 0
    if alpha <= 1.5: 
        R = 1
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
        n_real = n_out
    else: 
        R = 2
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2*(R+2))/2
        c0 = beta*(R-1)**3+1-c2
        n_real = 2*n_out

    #("long loop")

    n = (np.ceil(n_real*np.sqrt(2))).astype(int)
    m = (np.ceil(n_real*np.sqrt(2))).astype(int)
    tx = (np.arange(0,n)/n)*R
    ty = (np.arange(0,m)/m)*R

    rows = np.zeros((m,n))

    rs = np.sqrt( ( tx.reshape((n,1)) )**2 + (ty.reshape((1,m)))**2  )

    rows[ rs <= 1] = c0-rs[rs<=1]**alpha+c2*rs[rs<=1]**2
    rows[ (rs > 1 )& (rs <= R)] = beta*(R-rs[(rs>1)&(rs<=R)])**3/rs[(rs>1)&(rs<=R)]

    # ^ this might be hard to read, but it's just a np-vectorized
    # application of this function to the "rows" array:
    # def rho(p1, p2):
    #     # create continuous isotropic function
    #     r = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    #     if r <= 1:
    #         out=c0-r**alpha+c2*r**2
    #     elif r<=R:
    #         out=beta*(R-r)**3/r
    #     else:
    #         out=0
    #     return out

    #("stacking")

    loc_a1 = rows
    loc_a2 = rows[:, -1:1:-1]
    loc_b1 = rows[-1:1:-1, :]
    loc_b2 = rows[-1:1:-1 , -1:1:-1]

    loc_c1 = np.hstack([loc_a1, loc_a2])
    loc_c2 = np.hstack([loc_b1, loc_b2])
    block_circ = np.vstack([loc_c1, loc_c2])

    #("eigenvalues")

    # compute eigen-values
    lam = np.real(np.fft.fft2(block_circ))/(4*(m-1)*(n-1))
    lam = np.sqrt(lam)
    Z1 = rng.standard_normal((2*(m-1),2*(n-1)))
    Z2 = rng.standard_normal((2*(m-1),2*(n-1)))

    #("second fft2")

    # generate field with covariance given by block circular matrix
    Z = Z1 + 1j*Z2
    F = np.fft.fft2(lam * Z)
    #("generate field ")

    F = F[0:m,0:n] # extract sub-block with desired covariance
    # (out,c0,c2) = rho( (0,0) ,(0,0),R,2*H)
    field1 = np.real(F)
    field2 = np.imag(F) # two independent fields
    field1 = field1 - field1[0,0] # set field zero at origin
    field2 = field2 - field2[0,0] # set field zero at origin

    #("corrections")

    # make correction for embedding with a term c2*r^2

    loc1 = ty.conj().T *rng.standard_normal()
    loc2 = tx*rng.standard_normal()
    loc3 = np.kron(loc1, loc2).reshape((loc1.size, loc2.size))
    field1 = field1 + loc3*np.sqrt(2*c2)
    field2 = field2 + loc3*np.sqrt(2*c2)


    if alpha <= 1.5: 
        n_cut = int(n/(np.sqrt(2)))
        m_cut = int(m/(np.sqrt(2)))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]
    else:
        n_cut = int(n/(np.sqrt(2)*2))
        m_cut = int(m/(np.sqrt(2)*2))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]

    return field1



def fbm2D_exact_generator(H, n_out):

    rng = np.random.default_rng()

    alpha = 2*H

    #("set params")
    beta = 0
    c0 = 0
    c2 = 0
    R = 0
    n_real = 0
    if alpha <= 1.5: 
        R = 1
        beta = 0
        c2 = alpha/2
        c0 = 1-alpha/2
        n_real = n_out
    else: 
        R = 2
        beta = alpha*(2-alpha)/(3*R*(R**2-1))
        c2 = (alpha-beta*(R-1)**2*(R+2))/2
        c0 = beta*(R-1)**3+1-c2
        n_real = 2*n_out

    #("long loop")

    n = (np.ceil(n_real*np.sqrt(2))).astype(int)
    m = (np.ceil(n_real*np.sqrt(2))).astype(int)
    tx = (np.arange(0,n)/n)*R
    ty = (np.arange(0,m)/m)*R

    rows = np.zeros((m,n))

    rs = np.sqrt( ( tx.reshape((n,1)) )**2 + (ty.reshape((1,m)))**2  )

    rows[ rs <= 1] = c0-rs[rs<=1]**alpha+c2*rs[rs<=1]**2
    rows[ (rs > 1 )& (rs <= R)] = beta*(R-rs[(rs>1)&(rs<=R)])**3/rs[(rs>1)&(rs<=R)]

    # ^ this might be hard to read, but it's just a np-vectorized
    # application of this function to the "rows" array:
    # def rho(p1, p2):
    #     # create continuous isotropic function
    #     r = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    #     if r <= 1:
    #         out=c0-r**alpha+c2*r**2
    #     elif r<=R:
    #         out=beta*(R-r)**3/r
    #     else:
    #         out=0
    #     return out

    #("stacking")

    loc_a1 = rows
    loc_a2 = rows[:, -1:1:-1]
    loc_b1 = rows[-1:1:-1, :]
    loc_b2 = rows[-1:1:-1 , -1:1:-1]

    loc_c1 = np.hstack([loc_a1, loc_a2])
    loc_c2 = np.hstack([loc_b1, loc_b2])
    block_circ = np.vstack([loc_c1, loc_c2])

    #("eigenvalues")

    # compute eigen-values
    lam = np.real(np.fft.fft2(block_circ))/(4*(m-1)*(n-1))
    lam = np.sqrt(lam)

    # return generator parameters

    return (lam, n,m,tx,ty,c2,alpha )




def fbm2D_exact_from_generator(lam, n,m,tx,ty,c2,alpha ):
    rng = np.random.default_rng()

    Z1 = rng.standard_normal((2*(m-1),2*(n-1)))
    Z2 = rng.standard_normal((2*(m-1),2*(n-1)))

    #("second fft2")

    # generate field with covariance given by block circular matrix
    Z = Z1 + 1j*Z2
    F = np.fft.fft2(lam * Z)
    #("generate field ")

    F = F[0:m,0:n] # extract sub-block with desired covariance
    # (out,c0,c2) = rho( (0,0) ,(0,0),R,2*H)
    field1 = np.real(F)
    field2 = np.imag(F) # two independent fields
    field1 = field1 - field1[0,0] # set field zero at origin
    field2 = field2 - field2[0,0] # set field zero at origin

    #("corrections")

    # make correction for embedding with a term c2*r^2

    loc1 = ty.conj().T *rng.standard_normal()
    loc2 = tx*rng.standard_normal()
    loc3 = np.kron(loc1, loc2).reshape((loc1.size, loc2.size))
    field1 = field1 + loc3*np.sqrt(2*c2)
    field2 = field2 + loc3*np.sqrt(2*c2)


    if alpha <= 1.5: 
        n_cut = int(n/(np.sqrt(2)))
        m_cut = int(m/(np.sqrt(2)))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]
    else:
        n_cut = int(n/(np.sqrt(2)*2))
        m_cut = int(m/(np.sqrt(2)*2))

        field1 = field1[ 0:n_cut, 0:m_cut ]
        field2 = field2[ 0:n_cut, 0:m_cut ]

    return (field1, field2)


# gen_params = fbm2D_exact(0.7, 512)


# f1, f2 = fbm2D_exact_from_generator(*gen_params)
# plt.imshow(f1)
# plt.imshow(f2)


exact_image = fbm2D_exact(H=0.7, n_out=2**9)
plt.figure()
plt.imshow((exact_image))
plt.title("exact image")
plt.show()

midpoint_image = fract.fbm2D_midpoint(H=0.7, N=9)
plt.figure()
plt.imshow((midpoint_image))
plt.title("midpoint image")
plt.show()
