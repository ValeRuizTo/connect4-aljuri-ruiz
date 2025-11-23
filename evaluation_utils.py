
from training_env import train_agents
from aljuri_policy_trainable import ALjuriRuiz
from policyHello import HelloPolicy

# Crear 8 agentes inteligentes
agents_good = [ALjuriRuiz() for _ in range(8)]

# Crear 8 agentes aleapltorios
agents_random = [HelloPolicy() for _ in range(8)]

agents = agents_good + agents_random


# Para que todos compartan la misma memoria:
shared_Q = agents[0].Q
shared_N = agents[0].N
for a in agents:
    a.Q = shared_Q
    a.N = shared_N

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()   # importante para Windows
    train_agents(agents, episodes_per_pair=300)


# Guardar modelo final
agents[0].save("Q_table.json")
