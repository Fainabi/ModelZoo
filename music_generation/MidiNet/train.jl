include("model.jl")
include("preprocess.jl")

using JLD2
using ProgressMeter

Base.@kwdef struct Args
    lr::Float32    = 0.01
    epochs::Int    = 3
    batchsize::Int = 128
    device         = cpu
    truncate_size  = 10240
end

Base.@kwdef mutable struct Logger
    gen_num_sample::Int     = 0
    disc_num_sample::Int    = 0
    gen_loss::Float32   = 0.0f0
    disc_loss::Float32  = 0.0f0
    gen_acc::Float32    = 0.0f0
    disc_acc::Float32   = 0.0f0

    stop_training_disc  = false
end

function train()
    args = Args()

    # load dataset and dataloader
    if isfile("dataset.jld2")
        JLD2.@load "dataset.jld2" xs cond1d cond2d
    else
        xs, cond2d, cond1d = preprocess_dataset()
        JLD2.@save "dataset.jld2" xs cond1d cond2d
    end

    xs, cond1d, cond2d = Flux.DataLoader((xs, cond1d, cond2d), batchsize=args.truncate_size, shuffle=true) |> first
    dataloader = Flux.DataLoader((xs, cond1d, cond2d), batchsize=args.batchsize)

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
    opt_gen = ADAM(args.lr)
    opt_disc = ADAM(args.lr)
    for epoch in 1:args.epochs
        println("Epoch: ", epoch)

        # logger
        logger = Logger()
        p = Progress(length(dataloader))

        for (xs, cond1d, cond2d) in dataloader
            # train discriminator once but generator twice
            if !logger.stop_training_disc
                batch_size = size(cond1d, 2)
                logger.disc_num_sample += batch_size

                ps_disc = params(disc)

                zs = Flux.glorot_normal(100, batch_size)

                label = 1f0 - 0.0f0 * rand(Float32)  # label smoothing to [0.9, 1.0]

                loss, back = Flux.Zygote.pullback(ps_disc) do 
                    # predict for the true labels
                    y = disc(xs, cond1d)
                    y2 = disc(gen(zs, cond1d, cond2d), cond1d)

                    # compute accurancy
                    Flux.Zygote.ignore() do 
                        logger.disc_acc += (sum(sigmoid.(y) .>= 0.5) + sum(sigmoid.(y2) .<= 5)) / 2
                    end

                    # return loss
                    loss_fn(y, label) + loss_fn(y2, 0.0f0)
                end

                logger.disc_loss += loss * args.batchsize

                gs_disc = back(one(loss))
                Flux.update!(opt_disc, ps_disc, gs_disc)

                if logger.disc_acc / logger.disc_num_sample > 0.9
                    logger.stop_training_disc = true
                end
            end

            # update generator twice
            logger.gen_num_sample += args.batchsize * 2
            for _ in 1:2
                zs = Flux.glorot_normal(100, args.batchsize) |> args.device

                ps_gen = params(gen)

                loss, back = Flux.Zygote.pullback(ps_gen) do 
                    y = disc(gen(zs, cond1d, cond2d), cond1d)

                    Flux.Zygote.ignore() do 
                        logger.gen_acc += sum(sigmoid.(y) .>= 0.5)
                    end

                    loss_fn(y, 1.0f0)
                end

                logger.gen_loss += loss * args.batchsize

                gs_gen = back(one(loss))
                Flux.update!(opt_gen, ps_gen, gs_gen)
            end

            ProgressMeter.next!(p, showvalues = [(:gen_loss, logger.gen_loss / logger.gen_num_sample), 
                                                 (:gen_acc, logger.gen_acc / logger.gen_num_sample),
                                                 (:disc_loss, logger.disc_loss / logger.disc_num_sample), 
                                                 (:disc_acc, logger.disc_acc / logger.disc_num_sample)])

        end

        if epoch % 10 == 0
            @save string("models/snapshot", epoch, ".jld2") gen disc
        end
    end

    # download to cpu and store the model
    gen = cpu(gen)
    disc = cpu(disc)

    @save "model.jld2" gen disc
end
