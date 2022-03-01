MIN_MIDI_PATCH = 0
MAX_MIDI_PATCH = 128
NOTES_PER_OCTAVE = 12

DEFAULT_MIN_NOTE = 48
DEFAULT_MAX_NOTE = 84
DEFAULT_TRANSPOSE_TO_KEY = 0

struct EventSequenceRNNConfig
    hparams
    steps_per_quarter
    steps_per_second
    grid

    function EventSequenceRNNConfig(hparams, grid, steps_per_quarter=4, steps_per_second=100)
        hparams_dict = Dict(
            "batch_size" => 64,
            "rnn_layer_sizes" => [128, 128],
            "dropout_keep_prob" => 1.0,
            "attn_length" => 0,
            "clip_norm" => 3,
            "learning_rate" => 0.001,
            "residual_connections" => false,
            "use_cudnn" => false,

            "segment" => true,     # in the preprocessing step, whether to segment the notes
        )
        for (k, v) in hparams
            hparams_dict[k] == v
        end

        new(hparams_dict, steps_per_quarter, steps_per_second, grid)
    end
end


"""
    The structure of config imitates `MelodyRNNConfig` class in MelodyRNN.
"""
struct MelodyRNNConfig
    event_config::EventSequenceRNNConfig
    min_note
    max_note
    transpose_to_key

    function MelodyRNNConfig(event_config, min_n=DEFAULT_MIN_NOTE, max_n=DEFAULT_MAX_NOTE, t_key=DEFAULT_TRANSPOSE_TO_KEY)
        @assert min_n>=MIN_MIDI_PATCH "min_note must be >= 0, but was $min_n" 
        @assert max_n<=MAX_MIDI_PATCH "max_note must be <= 128, but was $max_n"
        @assert max_n-min_n >= NOTES_PER_OCTAVE string("interval between min_note and max_note must >= 12, ",
            "min_note: ", min_n, ", max_note: ", max_n)

        @assert (t_key>=0 && t_key<=NOTES_PER_OCTAVE - 1) "transpose_to_key must be >=0 or <= 11"
        
        new(event_config, min_n, max_n, t_key)
    end
end

default_configs = Dict(
    "basic_rnn" => MelodyRNNConfig(EventSequenceRNNConfig(Dict(), 0:1//4:1))
)

