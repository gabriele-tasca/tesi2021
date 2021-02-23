using Random, Distributions, PyPlot, DelimitedFiles
using PyCall
using Statistics
include("fbm1D.jl")

hs = pyimport("hurst")

N = 15

stat_n = 100

start = 0.05
stop = 0.9
step = 0.05
nhs = length(range(start,stop=stop,step=step))
res = zeros(Float64, nhs, 3)

for (i,H) in enumerate(range(start,stop=stop,step=step))
    H = start + i*step  
    stat_res = zeros(stat_n)
    for j in range(1, stop=stat_n) 

        vec = fbm1D(H, N=N)

        h_detected, c, data = hs.compute_Hc(vec, kind="random_walk", min_window=200 ,simplified=false)
        
        stat_res[j] = h_detected
    end
    av_h_detected = Statistics.mean(stat_res)
    std_h_detected = Statistics.std(stat_res)
    
    newrow = transpose([H, av_h_detected, std_h_detected])
    res[i,:] = newrow
end

print(res)
pygui(true)
errorbar(res[:,1], res[:,2], yerr=res[:,3]);
xlabel("generation H")
ylabel("detected H")
linex = [0,1]
liney = [0,1]
plot(linex, liney)
