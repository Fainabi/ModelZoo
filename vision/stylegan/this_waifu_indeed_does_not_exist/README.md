# This Waifu Indeed Does Not Exist

Amazing waifu ahead --> [This Waifu Does Not Exist(TWDNE)](https://www.thiswaifudoesnotexist.net/). And this repository is a julia implementation following Gwern's tutorial ([Making Anime Faces With StyleGAN](https://www.gwern.net/Faces)).

StyleGAN is based on ProGAN, and taking inspiration from _style transfer_. We omit to implementate ProGAN, but gave a style transfer implementation in the parent directory.

## Pretrained Model
Here we use [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl), together with the [`stylegan`](https://github.com/NVlabs/stylegan) tensorflow git to load existed model. 
To use `PyCall.jl` in conda, one needs to set the python's path, and rebuild the `PyCall.jl`, see [issue](https://github.com/JuliaPy/PyCall.jl/issues/598).