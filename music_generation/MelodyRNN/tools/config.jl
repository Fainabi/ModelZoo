import PyCall

function set_musescore_path(musescore)
    # cant simply write it in julia
    # which will cause a deep copy and not change the environment

PyCall.py"""
import music21

def set_musescore(path):
    us = music21.environment.UserSettings()

    us["musicxmlPath"] = path
    us["musescoreDirectPNGPath"] = path
"""

PyCall.py"set_musescore"(musescore)
end
