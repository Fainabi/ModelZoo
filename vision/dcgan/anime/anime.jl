using Flux
using Flux.Losses: logitbinarycrossentropy
using BSON
using Statistics
using Parameters: @with_kw
using Random
using ProgressBars: tqdm
using Zygote: pullback
using CUDA

CUDA.allowscalar(false)

# data preprocess
include("loaddata.jl")
data_path = joinpath(@__DIR__, "data/data.bson")
if !isfile(data_path)
    loaddata()
end


@with_kw struct Args
    lr_desc::Float32 = 2e-4
    lr_gen::Float32 = 2e-4
    device = gpu
    batchsize::Int = 16
    epochs::Int = 5000
    sampled_size = 6400
end

function Discriminator()
    Chain(
        Conv((5,5), 3=>32; stride=2, pad=1),
        BatchNorm(32, x->leakyrelu(x, 0.2f0)),
        # Dropout(0.25),  # first step of training can omit dropout layer
        Conv((5,5), 32=>64; stride=2, pad=1),
        BatchNorm(64, x->leakyrelu(x, 0.2f0)),
        # Dropout(0.25),
        Conv((4,4), 64=>128; stride=2, pad=1),
        BatchNorm(128, x->leakyrelu(x, 0.2f0)),
        # Dropout(0.25),
        Conv((4,4), 128=>256; stride=2, pad=1),
        BatchNorm(256, x->leakyrelu(x, 0.2f0)),
        # Dropout(0.25),
        x -> reshape(x, 3*3*256, :),
        Dense(3*3*256, 1)
    )
end

function Generator(hidden_dim)
    Chain(
        Dense(hidden_dim, 4*4*1024),
        x -> reshape(x, 4, 4, 1024, :),
        BatchNorm(1024, relu),
        ConvTranspose((5,5), 1024=>512; stride=2, pad=1),
        BatchNorm(512, relu),
        ConvTranspose((5,5), 512=>256; stride=2, pad=2),
        BatchNorm(256, relu),
        ConvTranspose((5,5), 256=>128; stride=2, pad=2),
        BatchNorm(128, relu),
        ConvTranspose((4,4), 128=>3, tanh; stride=2, pad=2)
    )
end

mutable struct AnimeGan
    generator
    discriminator
    AnimeGan(hidden_dim, args) = new(args.device(Generator(hidden_dim)), args.device(Discriminator()))
end
(m::AnimeGan)(xs) = m.discriminator(m.generator(xs))
sample(m::AnimeGan, xs) = m.generator(xs)

function fake!(m::AnimeGan, xs, opt_desc, opt_gen)
    ps_gen = params(m.generator)
    ps_desc = params(m.discriminator)

    loss_gen, back_gen = pullback(ps_gen) do 
        logitbinarycrossentropy(m.discriminator(m.generator(xs)), 1)
    end
    loss_desc, back_desc = pullback(ps_desc) do 
        logitbinarycrossentropy(m.discriminator(m.generator(xs)), 0)
    end

    Flux.update!(opt_gen, ps_gen, back_gen(one(loss_gen)))
    Flux.update!(opt_desc, ps_desc, back_desc(one(loss_desc)))
    

    loss_desc, loss_gen
end
function real!(m::AnimeGan, xs, opt_desc)
    ps = params(m.discriminator)

    loss, back = pullback(ps) do 
        logitbinarycrossentropy(m.discriminator(xs), 1)
    end
    gs = back(one(loss))
    Flux.update!(opt_desc, ps, gs)

    loss
end

function draw(m::AnimeGan, noise)
    image_arr = cpu(m.generator(noise))
    image_arr = vcat([image_arr[:, :, :, idx] for idx in 1:size(image_arr, 4)]...)
    image_arr = permutedims((image_arr .+ 1f0) ./ 2f0, [3,2,1]) |> colorview(RGB)
    image_arr
end

function train(;kws...)
    args = Args(kws...)
    device = args.device

    BSON.@load data_path data
    data = data[:, :, :, 1:args.sampled_size]
    data = 2*data .- 1.0f0
    dataloader = Flux.DataLoader(data, batchsize=args.batchsize, shuffle=true)
    latent_dim = 100

    m = AnimeGan(latent_dim, args)
    opt_g = ADAM(args.lr_gen)
    opt_d = ADAM(args.lr_desc)

    fixed_noises = device(randn(Float32, latent_dim, 10))

    @info "Start training"
    for epoch in 1:args.epochs
        loss_desc = 0
        loss_gan = 0
        for xs in tqdm(dataloader)
            loss_desc += real!(m, device(xs), opt_d)
            fake_desc, fake_gan = fake!(m, device(randn(Float32, latent_dim, args.batchsize)), opt_d, opt_g)
            loss_desc += fake_desc
            loss_gan += fake_gan
        end
        loss_desc /= length(dataloader)
        loss_gan /= length(dataloader)
        # no need for tracking accurancy here

        @info "Epoch $epoch:" loss_desc loss_gan
        save("pics/$epoch.png", draw(m, fixed_noises))
    end

    BSON.@save "model.bson" d=m.discriminator g=m.generator
end

train()