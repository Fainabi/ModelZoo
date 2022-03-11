using Flux
using Flux.Losses: logitbinarycrossentropy
using Flux.Zygote: pullback
using MLDatasets
using Statistics
using Random: randn
using Images
using ProgressBars


Flux.CUDA.allowscalar(false)
Base.@kwdef struct Args
    lr_disc::Float32 = 1e-3
    lr_gen::Float32 = 1e-3
    device = gpu
    batchsize::Int = 128
end

# This model imitates DCGAN in Flux's Model-Zoo
"A 28×28 to 2 network"
function Discriminator()
    Chain(
        Conv((3,3), 1=>64, x->leakyrelu(x, 0.2f0); stride=2, pad=1),  # or (4,4) for smaller size model
        Dropout(0.25),
        Conv((3,3), 64=>128, x->leakyrelu(x, 0.2f0); stride=2, pad=1),
        Dropout(0.25),
        x -> reshape(x, 7*7*128, :),
        Dense(7*7*128, 1)
    )
end

"latent_dim to (28, 28)"
function Generator(latent_dim)
    Chain(
        Dense(latent_dim, 7*7*256),
        BatchNorm(7*7*256, relu),
        x -> reshape(x, 7, 7, 256, :),
        ConvTranspose((5,5), 256=>128; stride=1, pad=2),
        BatchNorm(128, relu),
        ConvTranspose((4,4), 128=>64; stride=2, pad=1),
        BatchNorm(64, relu),
        ConvTranspose((4,4), 64=>1, tanh; stride=2, pad=1)
    )
end

mutable struct DCGAN
    generator
    discriminator
    DCGAN(latent_dim, args::Args) = begin
        new(args.device(Generator(latent_dim)), args.device(Discriminator()))
    end
end
(m::DCGAN)(xs) = m.discriminator(m.generator(xs))
sample(m::DCGAN, xs) = m.generator(xs)

function draw(m::DCGAN, noise)
    image_arr = cpu(m.generator(noise)) |> x -> dropdims(x; dims=3)
    image_arr = hcat([image_arr[:, :, idx] for idx in 1:size(image_arr, 3)]...)
    image_arr = @. Gray(image_arr + 1f0) / 2f0
    image_arr
end

function fake!(m::DCGAN, xs, opt_disc, opt_gen; args::Args)
    ps_gen = params(m.generator)
    ps_disc = params(m.discriminator)

    loss_gen, back_gen = pullback(ps_gen) do 
        logitbinarycrossentropy(m.discriminator(m.generator(xs)), 1)
    end
    loss_disc, back_disc = pullback(ps_disc) do 
        logitbinarycrossentropy(m.discriminator(m.generator(xs)), 0)
    end

    Flux.update!(opt_gen, ps_gen, back_gen(one(loss_gen)))
    Flux.update!(opt_disc, ps_disc, back_disc(one(loss_disc)))
    

    loss_disc, loss_gen
end
function real!(m::DCGAN, xs, opt_disc; args::Args)
    ps = params(m.discriminator)

    loss, back = pullback(ps) do 
        logitbinarycrossentropy(m.discriminator(xs), 1)
    end
    gs = back(one(loss))
    Flux.update!(opt_disc, ps, gs)

    loss
end

function train(; kws...)
    # args and device to use
    args = Args(; kws...)
    device = args.device

    # load MNIST dataset
    images, _ = MLDatasets.MNIST.traindata(Float32)  # images, _labels(discarded)
    images = device(images)
    images = 2*images .- 1                          # to [-1, 1]
    images = images |> x -> reshape(x, (28, 28, 1, :))
    @info "Size of image set" size(images)

    # construct dataloader
    dataloader = Flux.DataLoader(images, batchsize=args.batchsize, shuffle=true)
    latent_dim = 100
    m = DCGAN(latent_dim, args)
    opt_disc = ADAM(args.lr_disc)
    opt_gen = ADAM(args.lr_gen)

    # use these gaussian noises to generate new images
    fixed_noises = device(randn(Float32, latent_dim, 5))

    @info "Start Training..."
    for epoch in 1:5
        loss_disc = 0
        loss_gan = 0
        for xs in tqdm(dataloader)
            loss_disc += real!(m, xs, opt_disc; args)
            fake_disc, fake_gan = fake!(m, device(randn(Float32, latent_dim, args.batchsize)), opt_disc, opt_gen; args)
            loss_disc += fake_disc
            loss_gan += fake_gan
        end
        loss_disc /= length(dataloader)
        loss_gan /= length(dataloader)
        acc_real = mean(hcat((m.discriminator).(dataloader)...) .> 0.0f0)
        acc_fake = mean((m.discriminator∘m.generator)(device(randn(Float32, latent_dim, args.batchsize))) .< 0f0)

        @info "Epoch $epoch:" loss_disc loss_gan acc_real acc_fake
        save("pics/$epoch.png", draw(m, fixed_noises))
    end

end

train()