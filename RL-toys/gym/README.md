# Gym

Here we use [`PyCall`](https://github.com/JuliaPy/PyCall.jl) to interact with models in openai's [gym](https://github.com/openai/gym). The following models are assigned with agents to play:

- CartPole

One could create a gym environment in conda:

```bash
conda env create -f environment.yml
```

Such environment is named with `gym`. 

Then one can set the python and conda path in `config.jl`, and run

```julia
julia> ]activate .

julia> include("config.jl")
```

to rebuild the `PyCall` package.

