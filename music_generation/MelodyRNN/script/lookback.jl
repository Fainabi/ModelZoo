include("../models/config.jl")
include("../models/lookback.jl")

include("../tools/tokens.jl")
include("../tools/readfile.jl")
include("../tools/midi2vec.jl")
include("../tools/lookback.jl")

using JLD2
using Flux: onecold, Zygote
using ProgressMeter

Base.@kwdef struct KrnModelArgs
    lr::Float32     = 0.001
    hidden_dim::Int = 256
    epochs::Int     = 5
    seq_length::Int = 8
    batch_size::Int = 32
    device          = gpu

    num_songs::Int  = 20
    λ::Number       = 10f0
end

function train_lookback_model(dataset_path)
    args = KrnModelArgs()
    config = default_configs["basic_rnn"]

    # load dataset
    song_loader = FileReader(dataset_path)
    songs = collect(song_loader, args.num_songs)
    
    preprocessed_songs = []
    for song in songs
        tracks = extract_melody_tracks(song, config)

        preprocessed_songs = vcat(preprocessed_songs, tracks)
    end
    @assert (length(preprocessed_songs) > 0)  "No valid midi files detected"

    dataset, tokenset = generate_lookback_dataset(preprocessed_songs, args.seq_length, 4)
    xs, ys_note, ys_lookback = dataset
    xs = args.device.(xs)
    ys_note = args.device(ys_note)
    ys_lookback = args.device(ys_lookback)
    dataloader = Flux.DataLoader((xs..., ys_note, ys_lookback); batchsize=args.batch_size)
    
    # compile model
    model = construct_lookback_rnn(size(xs[1], 1), args.hidden_dim, length(tokenset))
    to_device(model, args.device)

    # train
    opt = ADAM(args.lr)
    loss_note = Flux.Losses.logitcrossentropy
    loss_lookback = Flux.Losses.logitbinarycrossentropy
    
    for epoch in 1:args.epochs
        println("Epoch: ", epoch)

        accumulated_losses = [0.0, 0.0]
        accumulated_acc = [0.0, 0.0]
        accumulated_size = [0]

        p = Progress(length(dataloader))

        # use for-loops to track accurancy and losses, thus note use default `Flux.train!` function
        for data in dataloader
            ps = params(model)

            # unpack data
            x, y_note, y_lookback = data[1:end-2], data[end-1], data[end]
            
            # reset rnn state
            Flux.reset!(model.rnn)
            
            # compute gradients
            loss, back = Zygote.pullback(ps) do 
                yhat_note, yhat_lookback = model(x)
                
                computed_loss_note = loss_note(yhat_note, y_note) 
                computed_loss_lookback = loss_lookback(yhat_lookback, y_lookback)

                # track loss and acc
                Zygote.ignore() do 
                    res_note = onecold(yhat_note, tokenset) .== onecold(y_note, tokenset)
                    res_lookback = map(yhat_lookback) do y
                        (sigmoid(y) >= 0.5) ? 1.0 : 0.0
                    end .== y_lookback

                    accumulated_size[1] += length(res_lookback)
                    accumulated_acc[1] += sum(res_note)
                    accumulated_acc[2] += sum(res_lookback)
                    accumulated_losses[1] += computed_loss_note * length(res_lookback)
                    accumulated_losses[2] += computed_loss_lookback * length(res_lookback)
                end

                # return loss
                computed_loss_lookback + computed_loss_note * args.λ
            end

            gs = back(one(loss))

            Flux.update!(opt, ps, gs)

            # print log
            ProgressMeter.next!(
                p; 
                showvalues=[(:loss_note, accumulated_losses[1] / accumulated_size[1]),
                            (:loss_lookback, accumulated_losses[2] / accumulated_size[1]),
                            (:acc_note, accumulated_acc[1] / accumulated_size[1]),
                            (:acc_lookback, accumulated_acc[2] / accumulated_size[1])]
            )
        end
    end

    # save model
    model = to_device(model, cpu)
    JLD2.@save "lookback.jld2" model tokenset seq_len=args.seq_length
end

train_lookback_model("dataset/clean_midi")

