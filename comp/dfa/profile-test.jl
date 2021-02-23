using PyCall
using PyPlot
using Statistics
using DelimitedFiles

pushfirst!(PyVector(pyimport("sys")."path"), "")
# Random.seed!(722229);


include("fbm2D.jl")

dfa = pyimport("dfa")
gen_pr = pyimport("gen-profile")

const N = 9

stat_n = 50

start = 0.15
stop = 0.9
step = 0.1
nhs = length(range(start,stop=stop,step=step))
res = zeros(Float64, nhs, 3)

bitmap = "./profiles/bitmap-9.png"

for (i,H) in enumerate(range(start,stop=stop,step=step))
    # H = start + i*step  
    stat_res = zeros(stat_n)
    println("i ",i," H ",H)
    for j in range(1, stop=stat_n) 

        # generate
        square_data = fbm2D(H,N=N)
        data = gen_pr.profile(square_data, bitmap)
        # PyPlot.imshow(data)
        # print(data)

        # dfa
        h_detected, c, scales, flucts = dfa.profile_dfa_from_z2d(data, s_max=Int(floor(side/20)))

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
writedlm("data/profiled-N="*string(N)*"-stat_n="*string(stat_n)*"-start="*string(start)*"-step="*string(step), res)

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

#########################

# generate
# H = 0.3
# N = 9



# square_data = fbm2D(H,N=N)
# using NPZ
# square_data = npzread("profiles/data-9.npy")
# bitmap = "./profiles/bitmap-9.png"


# data = gen_pr.profile(square_data, bitmap)

# # data = npzread("profiles/profile-10.npy")
# side = size(data)[1]

# # PyPlot.imshow(data)
# # print(data)

# # dfa
# h_detected, c, scales, flucts = dfa.profile_dfa_from_z2d(data, s_max=Int(floor(side/20)))