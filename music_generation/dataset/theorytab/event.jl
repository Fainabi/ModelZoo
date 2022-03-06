# This file contains events of music, 
# including 128 pitches from 0-127, rest event 128, continuous event 129, splitting token 255
# and bars splitting token 254

REST_TOKEN = 128            # show that at this beats start resting
CONTINUOUS_TOKEN = 129      # sustain that note for one unit
SPLITTING_TOKEN = 255       # show song starts or ends
BARS_TOKEN = 254            # show a measure ends

CENTER_C = 60               # MIDI number of C4

CHORD_REST_TOKEN = "R"
CHORD_CONTINUOUS_TOKE = "-"