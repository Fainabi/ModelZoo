include("../tools/dataset.jl")
include("../tools/krnreader.jl")
include("../models/lstm.jl")

using JLD2
using Flux: onecold, Zygote
using Statistics: mean
using ProgressMeter

Base.@kwdef struct KrnModelArgs
    lr::Float32     = 0.001
    hidden_dim::Int = 256
    epochs::Int     = 5
    seq_length::Int = 3
    batch_size::Int = 32
    device          = gpu
end

function train_model(dataset_paths)
    args = KrnModelArgs()

    # generate dataset
    songs = []

    for dataset_path in dataset_paths
        songs_read = load_dataset(dataset_path)
        songs_preprocessed = preprocess(songs_read)

        songs = vcat(songs, songs_preprocessed)
    end

    dataset, token_map = generate_training_sequences(songs, args.seq_length)
    xs, ys = dataset
    xs = args.device.(xs)  # arrays of matrix
    ys = args.device(ys)   # a single matrix

    dataloader = Flux.DataLoader((xs..., ys); batchsize=args.batch_size)

    # construct a neural network
    model = model_for_krn(length(token_map), args.hidden_dim)
    to_device(model, args.device)

    # train
    opt = ADAM(args.lr)
    loss_fn = Flux.Losses.logitcrossentropy

    for epoch in 1:args.epochs
        println("Epoch: ", epoch)

        accumulated_losses = 0.0
        accumulated_acc = [0.0]
        accumulated_size = [0]

        p = Progress(length(dataloader))
        for data in dataloader

            # compute gradient and update
            ps = params(model)  # get trainable parameters

            y = data[end]
            x = data[1:end-1]

            # reset rnn state
            # In batch training, different size of mini batch dataset may be constructed,
            # e.g.  32, 32, ..., 32, 10
            # and the last training step will cause `DimensionMismatch` error,
            # since the inner state will be tracked in Zygote, which has size of (seq_len, 32).
            # Thus here need to reset it first, before computing the gradients.
            Flux.reset!(model.rnn)

            loss, back = Zygote.pullback(ps) do 
                # compute predictions
                ŷ = model(x)

                # compute accurancy, ignore the gradients of computation
                Zygote.ignore() do 
                    acc_res = onecold(ŷ, token_map) .== onecold(y, token_map)
                    accumulated_size[1] += length(acc_res)
                    accumulated_acc[1] += sum(acc_res)
                end

                # return loss
                loss_fn(ŷ, y)
            end

            gs = back(one(loss))

            Flux.update!(opt, ps, gs)

            # log
            accumulated_losses += loss

            ProgressMeter.next!(p; showvalues=[(:loss, accumulated_losses / accumulated_size[1]),
                                                (:acc, accumulated_acc[1] / accumulated_size[1])])
        end
    end


    # save the model, parameters need loading on cpu
    to_device(model, cpu)
    @save "deutschl_model.jld2" model token_map
end

train_model(DEUTSUCHL_DATASETS)
