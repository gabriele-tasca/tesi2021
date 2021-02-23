
function get_RS(series)
    series = [12,65,546,47,243,12,76]
    incs = series[2:end] - series[1:end-1]  
    mean_inc = (series[end] - series[1])/length(incs)
    deviations = incs .- mean_inc
    Z = cumsum(deviations)
    R = maximum(Z) - minimum(Z)
    S = std(incs)
    if R == 0 || S == 0 ; return 0; end  # return 0 to skip this interval due undefined R/S
    return R / S
end


function compute_H(series; min_window=10, max_window=length(series)-1 , dt_step=0.25)
    window_sizes = range(log10(min_window), stop=log10(max_window), step=dt_step)
    # println("uefwioeg ",window_sizes)
    window_sizes = window_sizes .^10
    # println("uefwioeg 2222 ",window_sizes)

    RS = []
    for w in window_sizes
        rs = []
        for start in range(1, stop=len(series)-w, step=w)
            r = get_RS(series)
            if r != 0; append!(rs, r); end
        end
        append!(RS, mean(rs))
    end

    using PyCall
    linalg = pyimport("numpy.linalg")

    H = linalg.lstsq( (log10.(window_sizes), log10.(RS)) )
    println(H," ",c)
    return H, c
end



series = zeros(Float64, 100000)
using Random, Distributions
d = Normal(0,1)
for (i, e) in enumerate(series)
    series[i] = rand(d)
end
# println(series)
compute_H(series, min_window=100);