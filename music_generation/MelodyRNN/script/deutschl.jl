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

    dataloader = Flux.DataLoader(dataset; batchsize=args.batch_size)

    # construct a network
    model = model_for_krn(length(token_map), args.hidden_dim)

    # train
    opt = ADAM(args.lr)
    loss_fn = Flux.Losses.logitcrossentropy
    objective(x, y) = mean(@. loss_fn(model(x), y))

    evalcb() = begin
        x, y = dataset
        ŷ = map(x) do seq
            model(seq)
        end

        losses = mean(loss_fn.(ŷ, y))
        acc = map(ŷ, y) do pred, label
            onecold(pred, token_map) == onecold(label, token_map)
        end |> mean

        println("losses: ", losses, ", acc: ", acc)
    end

    for epoch in 1:1#args.epochs
        println("Epoch: ", epoch)

        accumulated_losses = 0.0
        accumulated_acc = [0.0]
        accumulated_size = [0]

        p = Progress(length(dataloader))
        for data in dataloader

            # compute gradient and update
            ps = params(model)

            x, y = data
            loss, back = Zygote.pullback(ps) do 
                # compute predictions
                ŷ = map(x) do seq
                    model(seq)
                end

                # compute accurancy
                Zygote.ignore() do 
                    acc_res = map(ŷ, y) do pred, label
                        onecold(pred, token_map) == onecold(label, token_map)
                    end
                    accumulated_size[1] += length(acc_res)
                    accumulated_acc[1] += sum(acc_res)
                end

                # return loss
                mean(loss_fn.(ŷ, y))
            end

            gs = back(one(loss))

            Flux.update!(opt, ps, gs)
            # Flux.train!(objective, params(model), dataloader, opt, cb=evalcb)


            # log
            accumulated_losses += loss

            ProgressMeter.next!(p; showvalues=[(:loss, accumulated_losses / accumulated_size[1]),
                                                (:acc, accumulated_acc[1] / accumulated_size[1])])
        end
    end


    # save and evaluate
    @save "deutschl_model.jld2" model token_map
end

train_model(DEUTSUCHL_DATASETS)
