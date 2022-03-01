# MelodyRNN

[MelodyRNN](https://github.com/magenta/magenta/tree/main/magenta/models/melody_rnn) is proposed by [magenta](https://github.com/magenta/magenta), a research project started by Google Brain team. It generates melody using RNN models, and thus produces so-called _Basic_, _Mono_, _Lookback_ and _Attention_ RNNs for melody generation.


## Models

We mainly implemented 

- Basic MelodyRNN (`models/lstm.jl`)
- Lookback MelodyRNN (`models/lookback.jl`)


For the basic one, one could go seeing the [video](https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz) to get basic music theory and LSTM based model construction. Since Mono MelodyRNN has exactly the same architecture to the basic one except the note ranging, we skiped that implmentation. 


## Dataset

We found [magenta](https://github.com/magenta/magenta/blob/main/magenta/scripts/README.md) and [Valerio Velardo](https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz) recommending [__Lakh MIDI Dataset__](http://colinraffel.com/projects/lmd/) and [deutschl](https://kern.humdrum.org/cgi-bin/browse?l=essen%2Feuropa%2Fdeutschl) krn dataset, so we mainly use these dataset for developing and testing. Some utilizable functions are written and put in `tools` directory.


## Train and Generate

We write some scripts for these models. They are in `script/` directory.

# Acknowledgement

The basic of melody generation can be learnt about with the videos in youtube from [Valerio Velardo](https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz). I was learning from that and trained a simple LSTM based melody generator.
