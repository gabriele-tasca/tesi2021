using Random, Distributions, PyPlot, DelimitedFiles

function power_of_two(exp::Int64) 2<<(exp-1) end

function fbm1D(H; N=18, base_stdev=1.0 )
    npoints = power_of_two(N) + 1;

    vec = zeros(Float64, npoints);
    g = 1::Int64
    for g in range(0, stop=N, step=1)
        gpow = power_of_two(g)
        gstep = (npoints-1) ÷ gpow
        # random addition
        stdev = (0.5)^(H*g) * base_stdev
        d = Normal(0, stdev)
        for s in range(1,stop=npoints,step=gstep)
            vec[s] += rand(d)
        end
        # interpolation
        if g != N
            for s in range( 1+gstep÷2, stop=npoints, step=gstep)
                left = s -(gstep ÷ 2)
                right = s + (gstep ÷ 2)
                vec[s] = (vec[left] + vec[right])/2
            end
        end
    end
    return vec;
end