import numpy as np
from connect4.policy import Policy


class HelloPolicy(Policy):

    def mount(self) -> None:
        pass

    def act(self, board: np.ndarray) -> int:

        def get_legal(b):
            return [c for c in range(7) if b[0, c] == 0]

        def apply_move(b, col, player):
            bb = b.copy()
            for r in range(5, -1, -1):
                if bb[r, col] == 0:
                    bb[r, col] = player
                    break
            return bb

        def winner_global(b):
            # Horizontal
            for r in range(6):
                for c in range(4):
                    line = b[r, c:c+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1

            # Vertical
            for c in range(7):
                col = b[:, c]
                for r in range(3):
                    line = col[r:r+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1

            # Diagonal derecha
            for r in range(3):
                for c in range(4):
                    diag = np.array([b[r+i, c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            # Diagonal izquierda
            for r in range(3):
                for c in range(3, 7):
                    diag = np.array([b[r+i, c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            if np.all(b[0] != 0):
                return 2
            return 0

        # -----------------------------
        # Comportamiento DEFENSIVO
        # -----------------------------
        legal = get_legal(board)

        # Determinar jugador oponente
        reds = np.sum(board == -1)
        yellows = np.sum(board == 1)
        opponent_player = -1 if reds == yellows else 1
        agent_player = -opponent_player

        # Si el OPONENTE puede ganar → bloquear
        for m in legal:
            bb = apply_move(board, m, opponent_player)
            if winner_global(bb) == opponent_player:
                return m

        #  Si YO puedo ganar → jugarlo (ataque simple)
        for m in legal:
            bb = apply_move(board, m, agent_player)
            if winner_global(bb) == agent_player:
                return m

        # Nada crítico → random
        return int(np.random.choice(legal))

    def learn(self, reward):
        pass  # Los agentes defensivos NO aprenden
