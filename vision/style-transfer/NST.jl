using Flux
using Flux.Losses: mse
using Metalhead
using Parameters
using Images
using Random: randn
using ProgressBars: tqdm

@with_kw struct Args
    β::Float32 = 1.
    epochs::Int = 5
    lr::Float32 = 2e-2
    device = cpu
end


function construct_model()
    vgg = VGG19()

    Chain(vgg.layers[1:21]...)
end

function content_loss(content_img, input_img)
    mse(content_img, input_img)
end

function gram(A)  # A is a tensor with several channels
    _, _, channels = size(A)
    A = reshape(A, :, channels)  # channel * channel
    A' * A
end

function style_loss(style_img, input_img)
    mse(gram(style_img), gram(input_img))
end

function extract_feature(model, img)
    img = reshape(img, size(img)..., 1)
    feature = model(img)
    dropdims(feature, dims=4)
end

function img_clip(img)
    map(img) do rgb
        if rgb < 0
            rgb = 0f0
        end
        if rgb > 1
            rgb = 1f0
        end
        rgb
    end
end

function train(pic1, pic2)
    args = Args()
    device = args.device
    vgg = construct_model() |> device
    content = float32.(load(pic1)) |> channelview |> x->permutedims(x, [3,2,1]) |> device
    style = float32.(load(pic2)) |> channelview |> x->permutedims(x, [3,2,1]) |> device

    new_pic = copy(content)
    content = extract_feature(vgg, content)
    style = extract_feature(vgg, style)
    
    opt = ADAM(args.lr)
    @info "Start Training..."
    for epoch in tqdm(1:args.epochs)
        ps = params(new_pic)
        gs = gradient(ps) do 
            feature = extract_feature(vgg, new_pic)
            content_loss(content, feature) + args.β*style_loss(style, feature)
        end
        Flux.update!(opt, ps, gs)
        new_pic = img_clip(new_pic)
    end

    new_img = permutedims(new_pic, [3,2,1]) |> colorview(RGB)
    save("target.jpg", new_img)
end
