using Random, Distributions, PyPlot, DelimitedFiles
using PyCall
include("fbm1D.jl")
include("hurst.jl")

vec = fbm1D(0.5, N=16)

pygui(true)
plot(vec)

H, c =compute_H(vec, min_window=100)