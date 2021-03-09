using Turing
using StatsPlots

@model function gdemo(x,y, ::Type{T} = Float64) where {T}
    if x === missing
        # Initialize `x` if missing
        #x is the state
        x = Vector{T}(undef, 10)
    end
    if y === missing
        # Initialize `x` if missing
        #y is the observation
        y = Vector{T}(undef, 10)
    end
    #s ~ InverseGamma(2, 3)
    #s ~ Normal(0, 1)
    #m ~ Normal(0, sqrt(s))
    for i in 1:length(x)#eachindex(x)
        #x[i] ~ Normal(m, sqrt(s))
        if i == 1
            #Define the initial distribution
            x[i] ~ Normal(0,1)
        else
            #Define the transition model
            x[i] ~ Normal(x[i-1]/2 + 25*x[i-1]/(1+x[i-1]^2)+8*cos(1.2*i),0.5^2)
        end
    end
    for i in 1:length(y)
        #Define the observation model
        y[i] ~ Normal(x[i]^2/20,0.5^2)
    end
end

# Construct a model with x = missing
model = gdemo(missing,missing)

#c1 = sample(model, SMC(), 1000)
# Replace num_chains below with however many chains you wish to sample.
chains = mapreduce(c -> sample(model, SMC(), 200), chainscat, 1)

#d = describe(c1)
#Plot the whole chain
plot(chains)
#Plot the distribution only
plot(chains, seriestype = :mixeddensity)

#plot(chains[:,[Symbol("x[$i]") for i in 1:10],:],seriestype = :mixeddensity)

#Plot the distribution at the first time step
plot(chains[:,[Symbol("x[1]")],:],seriestype = :mixeddensity)

# save the resultant states into a matrix
pf = zeros(10,200)
for t = 1:10
for i = 1:200
    pf[t,i] = copy(chains["x[$t]"].data[i])
end
end
# plot the states
plot(pf,legend=false)

## Following are some functions &examples for animation plots
using Plots
x = collect(1:0.1:30)
y = sin.(x)
df = 2

anim = @animate for i = 1:df:length(x)
    plot(x[1:i], y[1:i], legend=false)
end

gif(anim, "tutorial_anim_fps30.gif", fps = 30)

plot(pf[1,:])

anim = @animate for i = 1:df:10
    #plot(pf[:,i], legend=false)
    plot(chains[:,[Symbol("x[$i]")],:],seriestype = :mixeddensity)
end
gif(anim, "particle_filter.gif", fps = 30)

plot(pf)
