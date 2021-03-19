using Random, Distributions, PyPlot, DelimitedFiles

# Random.seed!(729);
function power_of_two(exp::Int64) 2<<(exp-1) end

function fbm2D(H; N = 4, base_stdev = 1.0)

    side = power_of_two(N) + 1
    npoints = side^2

    vec = zeros(Float64, side, side);
    g = 1::Int64
    for g in range(0, stop=N, step=1)
        gpow = power_of_two(g)
        gstep = (side-1) ÷ gpow
        # random addition
            stdev = (0.5)^(H*g) * base_stdev
        d = Normal(0, stdev)
        for i in range(1,stop=side,step=gstep)
            for j in range(1,stop=side,step=gstep)
                vec[i,j] += rand(d)
            end
        end
        # interpolation
        if g != N
            for i in range(1+gstep÷2, stop=side, step=gstep)
                for j in range(1, stop=side, step=gstep)
                    up = i - (gstep ÷ 2)   
                    down = i + (gstep ÷ 2)
                    vec[i,j] = (vec[up,j] + vec[down,j])/2
                end
            end

            for i in range(1, stop=side, step=gstep)
                for j in range(1+gstep÷2, stop=side, step=gstep)
                    left = j - (gstep ÷ 2)   
                    right = j + (gstep ÷ 2)
                    vec[i,j] = (vec[i,left] + vec[i,right])/2
                end
            end

            for i in range(1+gstep÷2, stop=side, step=gstep)
                for j in range(1+gstep÷2, stop=side, step=gstep)
                    up = j - (gstep ÷ 2)
                    down = j + (gstep ÷ 2)
                    left = i - (gstep ÷ 2)
                    right = i + (gstep ÷ 2)
                    vec[i,j] = (vec[i,up] + vec[i,down] + vec[left,j] + vec[right,j])/4
                end
            end
            # for s in range( 1+gstep÷2, stop=npoints, step=gstep)
            #     left = s -(gstep ÷ 2)
            #     right = s + (gstep ÷ 2)
            #     vec[s] = (vec[left] + vec[right])/2
            # end
        end
    end
    return vec
end

# const H = 0.3
# const n = 8
# side = power_of_two(n) + 1
# vec2 = fbm2D(H, N=n)

# using NPZ
# npzwrite("data-"*string(n)* ".npy", vec2)

# # writedlm( "frac2d.csv",  vec2, " ")
# pygui(true)
# p = PyPlot.imshow(vec2)
# PyPlot.colorbar(p)
# PyPlot.title("H = "* string(H) *" , "*string(side)*"x"*string(side)*" points")

