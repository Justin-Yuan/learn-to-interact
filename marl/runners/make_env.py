"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""
import os 
import sys 
mpe_hier_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    # top level 
mpe_hier_path = os.path.join(mpe_hier_path, "mpe_hierarchy")
sys.path.insert(1, mpe_hier_path)

# def make_env(scenario_name, benchmark=False, discrete_action=False):
#     '''
#     Creates a MultiAgentEnv object as env. This can be used similar to a gym
#     environment by calling env.reset() and env.step().
#     Use env.render() to view the environment on the screen.

#     Input:
#         scenario_name   :   name of the scenario from ./scenarios/ to be Returns
#                             (without the .py extension)
#         benchmark       :   whether you want to produce benchmarking data
#                             (usually only done during evaluation)

#     Some useful env properties (see environment.py):
#         .observation_space  :   Returns the observation space for each agent
#         .action_space       :   Returns the action space for each agent
#         .n                  :   Returns the number of Agents
#     '''
#     from multiagent.environment import MultiAgentEnv
#     import multiagent.scenarios as scenarios

#     # load scenario from script
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     if benchmark:        
#         env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                             scenario.observation, scenario.benchmark_data,
#                             discrete_action=discrete_action)
#     else:
#         env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                             scenario.observation,
#                             discrete_action=discrete_action)
#     return env



def make_env_hier(scenario_name, benchmark=False, show_visual_range=True, **kwargs):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from mpe_hierarchy.multiagent.environment import MultiAgentEnv
    import mpe_hierarchy.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(**kwargs)
    # create multiagent environment
    world_update_callback = getattr(scenario, "update_world", None)

    info_callback = None 
    # TODO: merge `benchmark data` with `info` in scenarios
    if benchmark:
        info_callback = scenario.benchmark_data
    # only set info_callback when available in scenario
    if hasattr(scenario, "info"):
        info_callback = scenario.info

    env = MultiAgentEnv(
        world, 
        scenario.reset_world, 
        scenario.reward, 
        scenario.observation,
        info_callback=info_callback,
        render_callback=scenario.render,
        update_callback=world_update_callback, 
        show_visual_range=show_visual_range
    )
    return env


#####################################################################################
### env mapping  
#####################################################################################

ENV_MAP = {
    "mpe_hier": make_env_hier
}