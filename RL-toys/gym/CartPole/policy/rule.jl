# rule based policy

struct RuleBasedPolicy <: AbstractPolicy end


function (π::RuleBasedPolicy)(env::CartPoleEnv)
    s = state(env)
    if s[3] > 0
        2
    else
        1
    end
end
