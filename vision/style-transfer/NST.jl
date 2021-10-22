using Flux
using Flux.Losses: mse
using Metalhead
using Parameters
using Images
using Random: randn

@with_kw struct Args
    β::Float32 = 1000000
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
    w, h, channels = size(A)
    A = reshape(A, :, channels)  # channel * channel
    A' * A / (w*h)^2
end

function style_loss(style_img, input_img)
    mse(gram(style_img), gram(input_img))
end

function extract_feature(model, img)
    features = []
    now_feature = img
    for layer in model.layers
        now_feature = layer(now_feature)
        push!(features, dropdims(now_feature, dims=4))
    end
    
    features
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

function img_preprocess(img)
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    img = (channelview(img) .- μ)./σ

    return Float32.(permutedims(img, (3, 2, 1))[:,:,:,:].*255)
end

function img_postprocess(img)
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    img = permutedims(img[:,:,:, 1] ./ 255, (3,2,1))
    img = (img .* σ) .+ μ

    return Float32.(img) |> img_clip |> colorview(RGB)
end

function train(pic1, pic2; kws...)
    args = Args(; kws...)
    device = args.device
    vgg = construct_model() |> device
    content = img_preprocess(load(pic1))
    style = img_preprocess(load(pic2))

    new_pic = copy(content)
    content = extract_feature(vgg, content)[end]
    style = extract_feature(vgg, style)
    
    opt = ADAM(args.lr)
    @info "Start Training..."
    for epoch in 1:args.epochs
        ps = params(new_pic)

        gs = gradient(ps) do 
            loss = 0
            feature = new_pic
            for (idx, layer) in enumerate(vgg.layers)
                feature = layer(feature)
                loss += args.β*style_loss(style[idx], feature)
            end
            loss + content_loss(content, feature)
        end
        Flux.update!(opt, ps, gs)

        features = extract_feature(vgg, new_pic)
        @info "Epoch: $epoch" content_loss(content, features[end]) style_loss(style[end], features[end])
    end

    new_img = img_postprocess(new_pic)
    save("target.jpg", new_img)
end
