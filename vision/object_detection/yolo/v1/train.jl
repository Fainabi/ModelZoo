include("model.jl")
include("dataset.jl")

using Flux
using JLD2
using ProgressMeter

Base.@kwdef struct Args
    lr::Float32  = 0.001f0
    device       = cpu
    batchsize    = 16
    epochs::Int  = 10
    load_model   = false
    load_model_file = "yolov1.jld2"
    img_dir      = "archive/images/"
    label_dir    = "archive/labels/"
    csv_file     = "archive/train.csv"
end

function construct_dataloader(args, idx=Colon(); test_csv=nothing)
    if isnothing(test_csv)
        imgs, labels = readdataset(args.csv_file, args.img_dir, args.label_dir; idx=idx)
    else
        imgs, labels = readdataset(test_csv, args.img_dir, args.label_dir; idx=idx)
    end
    imgs = args.device(imgs)
    labels = args.device(labels)
    dataloader = Flux.DataLoader((imgs, labels), batchsize=args.batchsize)
end

function train(model, dataloader, valid_dataloader, args::Args)
    loss_fn(x, y) = loss_yolov1(model(x), y)

    # logger
    p = Progress(length(dataloader))
    cb = () -> begin
        losses = 0
        acc    = 0

        for (xs, ys) in valid_dataloader
            ŷs = model(xs)

            losses += loss_yolov1(ŷs, ys)
        end

        ProgressMeter.next!(p, showvalues=[(:acc, acc), (:losses, losses)])
        # println("Loss: ", losses, ", acc: ", acc)
    end

    cb();

    for epoch in 1:args.epochs
        println("Epoch: ", epoch)
        Flux.train!(loss_fn, params(model), dataloader, ADAM(args.lr); cb = cb)
    end
end

function main()
    args = Args()

    # construct the yolo model
    yolov1 = FastYOLOv1()

    if args.load_model
        JLD2.@load args.load_model_file param
        Flux.loadparams!(yolov1, param)
    end
    yolov1 = args.device(yolov1)

    # construct dataset
    dataloader = construct_dataloader(args)
    validate_datalader = construct_dataloader(args; test_csv="archive/test.csv")

    # train
    train(yolov1, dataloader, validate_datalader, args)

    JLD2.@save args.load_model_file params(yolov1)

    return yolov1
end

