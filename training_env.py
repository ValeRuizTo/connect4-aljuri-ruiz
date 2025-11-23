import numpy as np
from connect4.connect_state import ConnectState

def play_game(agentA, agentB):
    """Juega una partida entre dos agentes y retorna el ganador."""
    agentA.mount()
    agentB.mount()

    state = ConnectState()
    history = []

    while not state.is_final():
        player = state.player

        if player == -1:
            action = agentA.act(state.board)
        else:
            action = agentB.act(state.board)

        history.append((state.board.copy(), action))
        state = state.transition(int(action))

    # Determinar ganador
    winner = state.get_winner()   # -1 (A), 1 (B), 2 draw, 0 error?

    # Recompensas desde el POV de cada agente
    if winner == -1:
        rewardA = +1
        rewardB = -1
    elif winner == 1:
        rewardA = -1
        rewardB = +1
    else:  # draw o 0
        rewardA = 0
        rewardB = 0

    # APRENDIZAJE AQUÍ
    agentA.learn(rewardA)
    agentB.learn(rewardB)

    return winner


def train_pair(args):
    agentA, agentB, episodes = args

    for _ in range(episodes):
        play_game(agentA, agentB)


def train_agents(agents, episodes_per_pair=200):
    import multiprocessing as mp

    tasks = []
    k = len(agents)

    for i in range(k):
        for j in range(i + 1, k):
            tasks.append((agents[i], agents[j], episodes_per_pair))

    print("\n=== ENTRENAMIENTO SECUENCIAL ===")
    total = len(tasks)
    count = 1

    for t in tasks:
        print(f"Entrenando par {count}/{total} ...")
        train_pair(t)
        count += 1

    print("Entrenamiento completado.")

