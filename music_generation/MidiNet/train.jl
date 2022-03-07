include("model.jl")
include("preprocess.jl")

using JLD2
using ProgressMeter

Base.@kwdef struct Args
    lr::Float32    = 0.01
    epochs::Int    = 3
    batchsize::Int = 128
    device         = cpu
end

Base.@kwdef mutable struct Logger
    num_sample::Int   = 0
    cum_loss::Float32 = 0.0f
end

function train()
    args = Args()

    # load dataset and dataloader
    xs, cond2d, cond1d = preprocess_dataset()
    dataloader = Flux.DataLoader((cs, cond1d, cond2d), batchsize=args.batchsize)

    # construct model
    gen = MidiNetGenerator()
    disc = MidiNetDiscriminator()

    # upload to gpu
    xs = args.device(xs)
    cond2d = args.device(cond2d)
    cond1d = args.device(cond1d)
    gen = args.device(gen)
    disc = args.device(disc)

    # loss function
    loss_fn = Flux.Losses.logitbinarycrossentropy

    # training step
    
    for epoch in 1:args.epochs
        println("Epoch: ", epoch)

        # logger
        p = Progress(length(dataloader))

        for (xs, cond1d, cond2d) in dataloader
            # train discriminator once but generator twice

            ps_disc = params(disc)
            loss, back = Flux.Zygote.pullback(ps_disc) do 
                y = disc(cond1d)

                Flux.Zygote.ignore() do 
                    
                end
            end

        end
    end

    # download to cpu and store the model
    gen = cpu(gen)
    disc = cpu(disc)

    @save "model.jld2" gen disc
end
