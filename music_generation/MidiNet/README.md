# MidiNet

MidiNet is a based on convolutional GAN, designed for symbolic domain generation. Thus it is still a language model. Origin paper can be found in [arxiv](https://arxiv.org/abs/1703.10847). Unlike the other models before, it introduces CNN into the generation, rather than the traditional RNN family. 

Respectively, DeepMind's [WaveNet](https://arxiv.org/abs/1609.03499v1) uses CNN to generate music in audio domain. Its typical sample-by-sample generating technique influence lots of the following papers in music generation and synthesis. 


## Model Design

### Representation

The traditional RNN models generate music note one by one, and that is so called continuously in MidiNet paper, in the symbolic domain. Different from that, MidiNet generates a bar at each time, thus the CNN technique can be applied. To match this idea, the representation is piano-roll like matrices. The matrices are binary, and every column is one-hot. Thus the matrix contains one dim for silence note, or rest.

Such representation cannot encode continous information of long duration notes.


### Convolutional GAN

MidiNet aims to dig more creativities, and thus uses framework of GAN to learn the distribution of latent noise space. Since the matrices representation is utilized, it is somehow the task like DCGAN for image generation, only the difference on the meaning of matrices. And the pipe line is:

$\mathcal{N}$ -> Deep Convolutional Generator -> Matrix of symbols in a bar -> Deep Convolutional Discriminator

### Conditional GAN

MidiNet adds more auxiliary structures to take attention on the previous bars of music notes, which makes it learn the structure of music. That is the novel conditional mechanism.

## Implementation

The implementation referred to the [pytorch implementation](https://github.com/annahung31/MidiNet-by-pytorch).

This deep NN model has complicated structure, since the condition information are injected into the data several times, which is similar to that of ResNet. And it may cause enormous compile time, see the [issue](https://github.com/FluxML/Zygote.jl/issues/1126). In such case, one can start the julia with `-O1` optimization, and reduce the compile time. If in the notebook or wanna adjust in the runtime, use `Base.Experimental.@optleval` to set it, see the [plot codes](https://github.com/JuliaPlots/Plots.jl/pull/2544/files) used.

