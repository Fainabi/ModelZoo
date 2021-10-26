using Flux
using Flux.Losses: mse
using Metalhead
using Parameters
using Images
using Statistics: mean, var
using Zygote: pullback

@with_kw struct Args
    lr::Float32 = 1f-5
    β::Float32 = 1000
    epochs::Int = 100
    device = gpu
end

function AdaIn(c, s)
    # WHCN
    c′ = (c .- mean(c, dims=(1,2))) ./ e_std(c, dims=(1,2))
    c′ .* e_std(s, dims=(1,2)) .+ mean(s, dims=(1,2))
end

function content_loss(x, y)
    mse(x, y) |> sqrt
end

function style_loss(t, s)  # at each layer
    sqrt(mse(mean(t, dims=(1,2)), mean(s, dims=(1,2)))) +
        sqrt(mse(e_std(t, dims=(1,2)), e_std(s, dims=(1,2))))
end

function extract_features(f, x)
    features = []
    feature = x
    for layer in f.layers
        feature = layer(feature)
        push!(features, feature)
    end
    
    features
end

function train(content_pic, style_pic; kws...)
    args = Args(; kws...)

    # preprocess
    encoder = VGG19().layers[1:12] |> args.device 
    decoder = Chain(map(reverse(encoder.layers)) do layer
        if layer isa MaxPool
            Upsample(scale=(2,2))
        else  # conv
            Conv(
                (3,3), 
                size(layer.weight, 4)=>size(layer.weight, 3),
                relu;
                pad=1  # consider reflection padding
            )
        end
    end...) |> args.device

    content = img_preprocess(load(content_pic)) |> args.device
    style = img_preprocess(load(style_pic)) |> args.device

    content_feature_maps = encoder(content)
    style_feature_maps = extract_features(encoder, style)

    t = AdaIn(content_feature_maps, style_feature_maps[end])

    # train
    opt = ADAM(args.lr)

    
    for epoch in 1:args.epochs
        ps = params(decoder)
        gs = gradient(ps) do 
            img_feature = decoder(t)
            loss = 0
            for (idx,layer) in enumerate(encoder.layers)
                img_feature = layer(img_feature)
                
                loss += style_loss(img_feature, style_feature_maps[idx])
            end
            # consider add content factor
            args.λ * loss + content_loss(img_feature, t)
        end
        
        Flux.update!(opt, ps, gs)
        img_feature = encoder(decoder(t))
        c_loss = content_loss(img_feature, t)
        s_loss_last = style_loss(img_feature, style_feature_maps[end])
        @info "Epoch $epoch:" c_loss s_loss_last
    end

    stylized_img = decoder(t) |> cpu
    save("target.jpg", img_postprocess(stylized_img))
end




# ===== Utils ====
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

function e_std(x; dims)
    ϵ::Float32 = 1e-2
    sqrt.(var(x, dims=dims) .+ ϵ)
end