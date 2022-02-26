# one 

DEUTSUCHL_DATASETS = map([
    "allerkbd",
    # "altdeu1",
    # "altdeu2",
    # "ballad",
    # "boehme",
    # "dva",
    # "erk",
    # "fink",
    # "kinder",
    # "test",
    # "variant",
    # "zuccal",
]) do subset
    joinpath(@__DIR__, "../dataset/deutschl", subset)
end
