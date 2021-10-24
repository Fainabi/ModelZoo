using PyCall
using Flux

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
end



mutable struct StyleBasedGen
    mapping
    synthesis
end

function Base.show(io::IO, gen::StyleBasedGen)
    print("8 FC Mapping + \nSynthesis(512")
    for syn in gen.synthesis
        print(" => ", size(syn.conv2.weight, 4))
    end
    println(")")
end

function StyleBasedGen(pyobj)
    tf_mapping = pyobj.components["mapping"].trainables
    tf_synthesis = pyobj.components["synthesis"].trainables


    # construct style gan
    mapping = Chain(map(0:7) do i
        weight = tf_mapping[string("Dense", i, "/weight")].eval()
        bias = tf_mapping[string("Dense", i, "/bias")].eval()
        layer = Dense(reverse(size(weight))...)
        Flux.loadparams!(layer, params(weight, bias))

        layer
    end...)

    synthesis = map(0:6) do i
        scale = 2^(i+2) |> x -> string(x, "x", x, "/")  # 8x8 to 256x256

        if i == 0
            const_head  = tf_synthesis[string(scale, "Const/const")].eval()
            # ? Const/bias
            noise1 = tf_synthesis[string(scale, "Const/Noise/weight")].eval()
            style1_w = tf_synthesis[string(scale, "Const/StyleMod/weight")].eval()
            style1_b = tf_synthesis[string(scale, "Const/StyleMod/bias")].eval()
            style1 = Dense(size(style1_w)...)  # (512, 1024)
            Flux.loadparams!(style1, params(style1_w', style1_b))

            conv2_w = tf_synthesis[string(scale, "Conv/weight")].eval()
            conv2_b = tf_synthesis[string(scale, "Conv/bias")].eval()
            conv2 = Conv((3,3), size(conv2_w, 3)=>size(conv2_w, 4))
            Flux.loadparams!(conv2, params(conv2_w, conv2_b))
            noise2 = tf_synthesis[string(scale, "Conv/Noise/weight")].eval()
            style2_w = tf_synthesis[string(scale, "Conv/StyleMod/weight")].eval()
            style2_b = tf_synthesis[string(scale, "Conv/StyleMod/bias")].eval()
            style2 = Dense(size(style2_w)...)
            Flux.loadparams!(style2, params(style2_w', style2_b))

            SynthesisNetwork(
                const_head, 
                noise1, 
                noise2, 
                style1, 
                style2, 
                nothing,
                conv2)
        else
            #? upsample 
            conv1_up_w = tf_synthesis[string(scale, "Conv0_up/weight")].eval()
            conv1_up_b = tf_synthesis[string(scale, "Conv0_up/bias")].eval()
            
            conv1_up = Conv((3,3), size(conv1_up_w, 3)=>size(conv1_up_w, 4))
            Flux.loadparams!(conv1_up, params(conv1_up_w, conv1_up_b))

            noise1 = tf_synthesis[string(scale, "Conv0_up/Noise/weight")].eval()
            style1_w = tf_synthesis[string(scale, "Conv0_up/StyleMod/weight")].eval()
            style1_b = tf_synthesis[string(scale, "Conv0_up/StyleMod/bias")].eval()
            style1 = Dense(size(style1_w)...)
            Flux.loadparams!(style1, params(style1_w', style1_b))


            conv2_w = tf_synthesis[string(scale, "Conv1/weight")].eval()
            conv2_b = tf_synthesis[string(scale, "Conv1/bias")].eval()
            conv2 = Conv((3,3), size(conv2_w, 3)=>size(conv2_w, 4))
            Flux.loadparams!(conv2, params(conv2_w, conv2_b))
            noise2 = tf_synthesis[string(scale, "Conv1/Noise/weight")].eval()
            style2_w = tf_synthesis[string(scale, "Conv1/StyleMod/weight")].eval()
            style2_b = tf_synthesis[string(scale, "Conv1/StyleMod/bias")].eval()
            style2 = Dense(size(style2_w)...)
            Flux.loadparams!(style2, params(style2_w', style2_b))


            SynthesisNetwork(
                nothing,
                noise1,
                noise2,
                style1,
                style2,
                conv1_up,
                conv2
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
