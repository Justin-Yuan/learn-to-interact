from agents.policy import DEFAULT_ACTION
from agents.ddpg import DDPGAgent
from agents.ppo import PPOAgent
from agents.sac import SACAgent
from agents.ddpg_moa import DDPGAgentMOA



#####################################################################################
### env mapping  
#####################################################################################

AGENTS_MAP = {
    "ddpg": DDPGAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
    "ddpg_moa": DDPGAgentMOA
}



__all__ = [
    "DEFAULT_ACTION",
    "DDPGAgent",
    "PPOAgent",
    "SACAgent",
    "DDPGAgentMOA"
]