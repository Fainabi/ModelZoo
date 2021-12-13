struct RandomPolicy <: AbstractPolicy end

(π::RandomPolicy)(env::AbstractEnv) = rand(action_space(env))
