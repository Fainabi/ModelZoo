using ReinforcementLearning

struct RecordHook <: AbstractHook
    rewards
    is_best::Array{Bool}
end

"""
Hook Acts on PreExperimentStage, PreEpisodeStage, PreActStage
         and PostExperimentStage, PostEpisodeStage, PostActStage
"""
