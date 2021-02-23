using PyCall
using PyPlot
using Statistics
using DelimitedFiles

Random.seed!(722229);


include("fbm2D.jl")

dfa = pyimport("./dfa.py")

const N = 9

stat_n = 50

start = 0.1
stop = 0.9
step = 0.1
nhs = length(range(start,stop=stop,step=step))
res = zeros(Float64, nhs, 3)

for (i,H) in enumerate(range(start,stop=stop,step=step))
    # H = start + i*step  
    stat_res = zeros(stat_n)
    println("i ",i," H ",H)
    for j in range(1, stop=stat_n) 

        # generate
        # Random.seed!(729 + i*34);
        data = fbm2D(H,N=N)

        # dfa
        h_detected, c, scales, flucts = dfa.square_grid_dfa_from_z2d(data)

        stat_res[j] = h_detected
        println("     ", j)
    end
    av_h_detected = Statistics.mean(stat_res)
    std_h_detected = Statistics.std(stat_res)
    
    newrow = transpose([H, av_h_detected, std_h_detected])
    res[i,:] = newrow
end

# print(res)
pygui(true)
errorbar(res[:,1], res[:,2], yerr=res[:,3]);
xlabel("generation H")
ylabel("detected H")
linex = [0,1]
liney = [0,1]
plot(linex, liney)
# writedlm("data/data-N="*string(N)*"-stat_n="*string(stat_n)*"-start="*string(start), res)

# 1 shot

# H = 0.4
# N = 8
# # generate
# data = fbm2D(H,N=N)

# # dfa
# h_detected, c, scales, flucts = dfa.square_grid_dfa_from_z2d(data)


# plt.scatter(scales,flucts, marker=".", color="deepskyblue")
# plt.plot(scales, (10^c)*scales.^H, color="purple")
# plt.xscale("log")
# plt.yscale("log")
# plt.title("H = " * string(H) * ", detected = "* string(h_detected))
# plt.show()
