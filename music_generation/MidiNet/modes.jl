KEYS = Dict(
    "C" => 0,
    "C#" => 1,
    "Db" => 1,
    "D" => 2,
    "D#" => 3,
    "Eb" => 3,
    "E" => 4,
    "F" => 5,
    "F#" => 6,
    "Gb" => 6,
    "G" => 7,
    "G#" => 8,
    "Ab" => 8,
    "A" => 9,
    "A#" => 10,
    "Bb" => 10,
    "B" => 11,
)

PITCH_KEY = Dict(map(reverse, collect(KEYS)))


# According to the cheet sheet (https://www.hooktheory.com/cheat-sheet),
#  hooktheory uses the 7 modern modes (https://en.wikipedia.org/wiki/Mode_(music)#Modern_modes)
#  Ionian (major scale), Dorian, Phrygian, Lydian, Mixolydian, Aeolian (minor scale), and Locrian.
#  Since the songs are all popular songs, we do not consider pentatonic scales or chromatic scale.


Ionian = [0, 2, 4, 5, 7, 9, 11]
Dorian = [0, 2, 3, 5, 7, 9, 10]
Phrygian = [0, 1, 3, 5, 7, 8, 10]
Lydian = [0, 2, 4, 6, 7, 9, 11]
Mixolydian = [0, 2, 4, 5, 7, 9, 10]
Aeolian = [0, 2, 3, 5, 7, 8, 10]
Locrian = [0, 1, 3, 5, 6, 8, 10]
HarmonicMinor = [0, 2, 3, 5, 7, 8, 11]
PhrygianDominant = [0, 1, 4, 5, 7, 8, 10]


Ionian_5th_chords = ["M", "m", "m", "M", "M", "m", "∘"]
Dorian_5th_chords = ["m", "m", "M", "M", "m", "∘", "M"]
Phrygian_5th_chords = ["m", "M", "M", "m", "∘", "M", "m"]
Lydian_5th_chords = ["M", "M", "m", "∘", "M", "m", "m"]
Mixolydian_5th_chords = ["M", "m", "∘", "M", "m", "m", "M"]
Aeolian_5th_chords = ["m", "∘", "M", "m", "m", "M", "M"]
Locrian_5th_chords = ["∘", "M", "m", "m", "M", "M", "m"]
HarmonicMinor_5th_chords = ["m", "∘", "+", "m", "M", "M", "∘"]
PhrygianDominant_5th_chords = ["M", "M", "∘", "m", "∘", "+", "m"]


Ionian_7th_chords = ["M7", "m7", "m7", "M7", "7", "m7", "∅7"]
Dorian_7th_chords = ["m7", "m7", "M7", "7", "m7", "∅7", "M7"]
Phrygian_7th_chords = ["m7", "M7", "7", "m7", "∅7", "M7", "m7"]
Lydian_7th_chords = ["M7", "7", "m7", "∅7", "M7", "m7", "m7"]
Mixolydian_7th_chords = ["7", "m7", "∅7", "M7", "m7", "m7", "M7"]
Aeolian_7th_chords = ["m7", "∅7", "M7", "m7", "m7", "M7", "7"]
Locrian_7th_chords = ["∅7", "M7", "m7", "m7", "M7", "7", "m7"]
HarmonicMinor_7th_chords = ["mM7", "∅7", "+M7", "m7", "7", "M7", "∘7"]
PhrygianDominant_7th_chords = ["7", "M7", "∘7", "mM7", "∅7", "+M7", "m7"]


Ionian_9th_chords = ["M9", "m9", "m9", "M9", "9", "m9", "m7b5b9"]
Dorian_9th_chords = ["m9", "m9", "M9", "9", "m9", "m7b5b9", "M9"]
Phrygian_9th_chords = ["m9", "M9", "9", "m9", "m7b5b9", "M9", "m9"]
Lydian_9th_chords = ["M9", "9", "m9", "m7b5b9", "M9", "m9", "m9"]
Mixolydian_9th_chords = ["9", "m9", "m7b5b9", "M9", "m9", "m9", "M9"]
Aeolian_9th_chords = ["m9", "m7b5b9", "M9", "m9", "m9", "M9", "9"]
Locrian_9th_chords = ["m7b5b9", "M9", "m9", "m9", "M9", "9", "m9"]
HarmonicMinor_9th_chords = ["mM9", "m7b5b9", "M9#5", "m9", "9", "M7#9", "∘7b9"]
PhrygianDominant_9th_chords = ["9", "M7#9", "∘7b9", "mM9", "m7b5b9", "M9#5", "m9"]

# the 11th chords and 13th chords are complicated
# and at the preprocessing step, these chords will be changed into 9th chords.



MODES = Dict(
    "major" => Ionian,
    "dorian" => Dorian,
    "phrygian" => Phrygian,
    "lydian" => Lydian,
    "mixolydian" => Mixolydian,
    "minor" => Aeolian,
    "locrian" => Locrian,
    "harmonicMinor" => HarmonicMinor,
    "phrygianDominant" => PhrygianDominant,

    1 => Ionian,
    2 => Dorian,
    3 => Phrygian,
    4 => Lydian,
    5 => Mixolydian,
    6 => Aeolian,
    7 => Locrian,
)

CHORDS = Dict(
    5 => Dict(
        "major"             => Ionian_5th_chords,
        "dorian"            => Dorian_5th_chords,
        "phrygian"          => Phrygian_5th_chords,
        "lydian"            => Lydian_5th_chords,
        "mixolydian"        => Mixolydian_5th_chords,
        "minor"             => Aeolian_5th_chords,
        "locrian"           => Locrian_5th_chords,
        "harmonicMinor"     => HarmonicMinor_5th_chords,
        "phrygianDominant"  => PhrygianDominant_5th_chords
    ),
    7 => Dict(
        "major"             => Ionian_7th_chords,
        "dorian"            => Dorian_7th_chords,
        "phrygian"          => Phrygian_7th_chords,
        "lydian"            => Lydian_7th_chords,
        "mixolydian"        => Mixolydian_7th_chords,
        "minor"             => Aeolian_7th_chords,
        "locrian"           => Locrian_7th_chords,
        "harmonicMinor"     => HarmonicMinor_7th_chords,
        "phrygianDominant"  => PhrygianDominant_7th_chords
    ),
    9 => Dict(
        "major"             => Ionian_9th_chords,
        "dorian"            => Dorian_9th_chords,
        "phrygian"          => Phrygian_9th_chords,
        "lydian"            => Lydian_9th_chords,
        "mixolydian"        => Mixolydian_9th_chords,
        "minor"             => Aeolian_9th_chords,
        "locrian"           => Locrian_9th_chords,
        "harmonicMinor"     => HarmonicMinor_9th_chords,
        "phrygianDominant"  => PhrygianDominant_9th_chords
    ),
)

