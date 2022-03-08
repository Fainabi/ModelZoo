using Flux
using Flux: @functor


mutable struct MidiNetGenerator
    mlp
    conv1
    conv2
    conv3
    conv4

    conv_tran1
    conv_tran2
    conv_tran3
    conv_tran4

end

function MidiNetGenerator() 
    # for generator 
    latent_dim = 100
    cond_1d_dim = 13
    cond_2d_dim = 16
    pitch_range = 128

    mlp = Chain(
        Dense(latent_dim + cond_1d_dim, 1024),
        BatchNorm(1024, relu),
        Dense(1024, 256),  # the paper used 512, and 1d-dimention condition is injected twice
        BatchNorm(256, relu)
    )

    conv_trans = map(1:4) do i
        in_channel = pitch_range + cond_2d_dim + cond_1d_dim
        stride = (i == 4) ? (2, 1) : (2, 2)
        dimpair = (i < 4) ? (in_channel=>pitch_range) : (in_channel=>1)
        kernel_size = (i < 4) ? (1, 2) : (pitch_range, 1)
        batchnorm_size = (i == 4) ? 1 : pitch_range

        Chain(
            ConvTranspose(kernel_size, dimpair, stride=stride),
            BatchNorm(batchnorm_size, relu),
        )
    end

    # for conditioner
    convs = map(1:4) do i
        kernel_size = (i == 1) ? pitch_range : 1
        stride = (i == 1) ? (2, 1) : (2, 2)
        dimpair = (i == 1) ? (1 => 16) : (16 => 16)
        
        Chain(
            Conv((kernel_size, 1), dimpair, stride=stride),
            BatchNorm(16, x -> leakyrelu(x, 0.2)),
        )
    end

    MidiNetGenerator(mlp, convs..., conv_trans...)
end

@functor MidiNetGenerator

Flux.params(gen::MidiNetGenerator) = Flux.params([
    gen.mlp,
    gen.conv1,
    gen.conv2,
    gen.conv3,
    gen.conv4,
    gen.conv_tran1,
    gen.conv_tran2,
    gen.conv_tran3,
    gen.conv_tran4
]...)

function (gen::MidiNetGenerator)(z, cond_1d, cond_2d)
    # size(z) => (100, batch_num)
    # size(cond_1d) => (cond_1d_dim, batch_num)
    # size(cond_2d) => (pitch_range, notes_in_a_bar, batch_num)

    z = vcat(z, cond_1d)  # (100+cond_1d_dim, batch_num)
    z_proj = gen.mlp(z)  # (256, batch_num)

    # compute the conditions
    cond_2d = reshape(cond_2d, 128, 16, 1, :)  # magic number for pitch_range and the bar length over notes
    cond1 = gen.conv1(cond_2d)
    cond2 = gen.conv2(cond1)
    cond3 = gen.conv3(cond2)
    cond4 = gen.conv4(cond3)


    # tranposed convolution
    h = reshape(z_proj, 1, 2, 128, :)  # (128, 2, 1, batch_num)
    
    cond_1d = reshape(cond_1d, 1, 1, 13, :)
    cond_1d = cat(cond_1d, cond_1d; dims=2)  # (13, 2, 1, batch_num)
    h = cat(h, cond_1d, cond4; dims=3)
    h = gen.conv_tran1(h)

    cond_1d = cat(cond_1d, cond_1d; dims=2)
    h = cat(h, cond_1d, cond3; dims=3)
    h = gen.conv_tran2(h)

    cond_1d = cat(cond_1d, cond_1d; dims=2)
    h = cat(h, cond_1d, cond2; dims=3)
    h = gen.conv_tran3(h)

    cond_1d = cat(cond_1d, cond_1d; dims=2)
    h = cat(h, cond_1d, cond1; dims=3)
    h = gen.conv_tran4(h)

    # h = sigmoid.(h)  # may cause gradient vanishment?
    h = softmax(h)

    dropdims(h; dims=3)
end




mutable struct MidiNetDiscriminator
    mlp
    conv1
    conv2
end

function MidiNetDiscriminator()
    pitch_range = 128

    mlp = Chain(
        Dense(244, 1024),
        BatchNorm(1024, relu),
        Dense(1024, 1)
    )
    MidiNetDiscriminator(
        mlp,
        # no cond_2d input here
        Conv((pitch_range, 2), 14 => 14, x -> leakyrelu(x, 0.2); stride=2),
        Chain(
            Conv((1, 4), 27 => 77; stride=(2,2)),
            BatchNorm(77, x -> leakyrelu(x, 0.2))
        )
    )
end

@functor MidiNetDiscriminator

Flux.params(disc::MidiNetDiscriminator) = Flux.params(disc.mlp, disc.conv1, disc.conv2)

function (disc::MidiNetDiscriminator)(input, cond_1d)
    x = reshape(input, 128, 16, 1, :)
    # cond_2d = reshape(cond_2d, 128, 16, 1, :)
    cond = vec(cond_1d)
    cond = vcat(fill(cond, 128*16)...)
    cond = reshape(cond, 13, :, 128, 16)
    cond = permutedims(cond, (3, 4, 1, 2))

    x = cat(x, cond; dims=3)
    x = disc.conv1(x)

    cond = vec(cond_1d)
    cond = vcat(fill(cond, 8)...)
    cond = reshape(cond, 13, :, 1, 8)
    cond = permutedims(cond, (3, 4, 1, 2))

    x = cat(x, cond; dims=3)
    x = disc.conv2(x)

    x = reshape(x, 231, :)
    x = cat(x, cond_1d; dims=1)

    disc.mlp(x)
end



