using ReinforcementLearning

function RLCore.run(
    policy::AbstractPolicy,
    env::AbstractEnv,
    stop_condition = StopAfterEpisode(1),
    hook = EmptyHook(),
    noreset=false,
)
    RLCore.check(policy, env)
    RLCore._run(policy, env, stop_condition, hook; noreset=noreset)
end

function RLCore._run(policy::AbstractPolicy, env::AbstractEnv, stop_condition, hook::AbstractHook; noreset=false)

    hook(PRE_EXPERIMENT_STAGE, policy, env)
    policy(PRE_EXPERIMENT_STAGE, env)
    is_stop = false
    while !is_stop
        if !noreset
            reset!(env)
        else
            noreset = false
        end
        policy(PRE_EPISODE_STAGE, env)
        hook(PRE_EPISODE_STAGE, policy, env)

        while !is_terminated(env) # one episode
            action = policy(env)

            policy(PRE_ACT_STAGE, env, action)
            hook(PRE_ACT_STAGE, policy, env, action)

            env(action)

            policy(POST_ACT_STAGE, env)
            hook(POST_ACT_STAGE, policy, env)

            if stop_condition(policy, env)
                is_stop = true
                break
            end
        end # end of an episode

        if is_terminated(env)
            policy(POST_EPISODE_STAGE, env)  # let the policy see the last observation
            hook(POST_EPISODE_STAGE, policy, env)
        end
    end
    hook(POST_EXPERIMENT_STAGE, policy, env)
    hook
end

