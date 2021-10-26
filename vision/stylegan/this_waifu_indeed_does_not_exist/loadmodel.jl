using PyCall
using Flux
using Random: randn

include("utils.jl")

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), "")
    nowpath = @__DIR__
    rootpath = joinpath(replace(@__FILE__, r"loadmodel.jl$" => ""))

    cd(joinpath(rootpath, "stylegan"))

    py"""
    import io
    import pickle
    import dnnlib.tflib as tflib



    def get_model(filename):
        tflib.init_tf()

        with open(filename, 'rb') as f:
            G, D, Gs = pickle.load(io.BytesIO(f.read()))

        return G, D, Gs
    """

    cd(nowpath)
end



mutable struct SynthesisNetwork
    const_head

    noise1
    noise2
    style1
    style2
    conv1_up
    conv2

    upsample
end



mutable struct StyleBasedGen
    mapping
    synthesis
end

"""
    generate_sample(w, noise)

#params
...
w: style sample W space
noise: sample in gaussian noises
...
"""
function generate_sample(gen::StyleBasedGen, w, noise=nothing)
    x = gen.synthesis[1].const_head
    for (idx, layer) in enumerate(gen.synthesis)
        if idx > 1
            x = x |> layer.upsample |> layer.conv1_up
        end
        noise1 = isnothing(noise) ? randn() : noise[idx][1]
        noise2 = isnothing(noise) ? randn() : noise[idx][2]
        x += layer.noise1 * noise1
        x = AdaIn(layer.style1(w), x)
        x = layer.conv2(x) + layer.noise2 * noise2
        x = AdaIn(layer.style2(w), x)
    end
end



function Base.show(io::IO, gen::StyleBasedGen)
    print("8 FC Mapping + \nSynthesis(")
    for syn in gen.synthesis
        if syn.conv1_up !== nothing
            print("(", size(syn.conv1_up.weight, 3), ",", size(syn.conv1_up.weight, 4), "), ")
        end
        println("(", size(syn.conv2.weight, 3), ",", size(syn.conv2.weight, 4), "), ")
    end
    println(")")
end

function load_conv(tf_conv_w, tf_conv_b)
    cn = Conv((3,3), size(tf_conv_w, 3)=>size(tf_conv_w, 4); pad=1)
    Flux.loadparams!(cn, params(tf_conv_w, tf_conv_b))

    cn
end

function load_fc(tf_w, tf_b, activation=identity)
    fc = Dense(size(tf_w)..., activation)
    Flux.loadparams!(fc, params(tf_w', tf_b))

    fc
end

function StyleBasedGen(pyobj)
    tf_mapping = pyobj.components["mapping"].trainables
    tf_synthesis = pyobj.components["synthesis"].trainables


    # construct style gan
    mapping = Chain(map(0:7) do i
        load_fc(
            tf_mapping[string("Dense", i, "/weight")].eval(),
            tf_mapping[string("Dense", i, "/bias")].eval(),
            x -> leakyrelu(x, 0.2)
        )
    end...)

    synthesis = map(0:6) do i
        scale = 2^(i+2) |> x -> string(x, "x", x, "/")  # 8x8 to 256x256

        if i == 0
            const_head  = tf_synthesis[string(scale, "Const/const")].eval()
            # ? Const/bias
            noise1 = tf_synthesis[string(scale, "Const/Noise/weight")].eval()
            style1 = load_fc(
                tf_synthesis[string(scale, "Const/StyleMod/weight")].eval(),
                tf_synthesis[string(scale, "Const/StyleMod/bias")].eval()
            )

            conv2 = load_conv(
                tf_synthesis[string(scale, "Conv/weight")].eval(), 
                tf_synthesis[string(scale, "Conv/bias")].eval())
            noise2 = tf_synthesis[string(scale, "Conv/Noise/weight")].eval()
            style2 = load_fc(
                tf_synthesis[string(scale, "Conv/StyleMod/weight")].eval(),
                tf_synthesis[string(scale, "Conv/StyleMod/bias")].eval()
            )

            SynthesisNetwork(
                const_head, 
                noise1, 
                noise2, 
                style1, 
                style2, 
                nothing,
                conv2, 
                nothing
            )
        else
            conv1_up = load_conv(
                tf_synthesis[string(scale, "Conv0_up/weight")].eval(),
                tf_synthesis[string(scale, "Conv0_up/bias")].eval()
            )
            noise1 = tf_synthesis[string(scale, "Conv0_up/Noise/weight")].eval()
            style1 = load_fc(
                tf_synthesis[string(scale, "Conv0_up/StyleMod/weight")].eval(),
                tf_synthesis[string(scale, "Conv0_up/StyleMod/bias")].eval()
            )


            conv2 = load_conv(
                tf_synthesis[string(scale, "Conv1/weight")].eval(),
                tf_synthesis[string(scale, "Conv1/bias")].eval()
            )
            noise2 = tf_synthesis[string(scale, "Conv1/Noise/weight")].eval()
            style2 = load_fc(
                tf_synthesis[string(scale, "Conv1/StyleMod/weight")].eval(),
                tf_synthesis[string(scale, "Conv1/StyleMod/bias")].eval()
            )

            SynthesisNetwork(
                nothing,
                noise1,
                noise2,
                style1,
                style2,
                conv1_up,
                conv2,
                Upsample(:bilinear, size=(2,2))
            )
        end
    end

    StyleBasedGen(mapping, synthesis)
end

function loadmodel(name)
    __init__()

    G, D, Gs = py"get_model"(name)

    G = StyleBasedGen(G)
    Gs = StyleBasedGen(Gs)
end
