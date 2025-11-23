import numpy as np
from connect4.policy import Policy

# MCTS CONNECT-4 POLICY – ALJURI-RUIZ

class ALjuriRuiz(Policy):

    def mount(self, time_out: int)-> None:
        pass

    def act(self, board: np.ndarray) -> int:

        #   CONFIGURACIÓN MCTS
        ITER = 50
        C = 1.41

        #   FUNCIONES INTERNAS
    
        def infer_player(b):
            reds = np.sum(b == -1)
            yellows = np.sum(b == 1)
            return -1 if reds == yellows else 1

        def get_legal(b):
            return [c for c in range(7) if b[0, c] == 0]

        def apply_move(b, col, player):
            newb = b.copy()
            for r in range(5, -1, -1):
                if newb[r, col] == 0:
                    newb[r, col] = player
                    break
            return newb

        def winner(b):
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

            # Empate
            if np.all(b[0] != 0):
                return 2

            return 0


        #  CHECKS TÁCTICOS: WIN / BLOCK

        def immediate_tactics(b, player, legal):
            # 1. WIN if possible
            for m in legal:
                nb = apply_move(b, m, player)
                if winner(nb) == player:
                    return m

            # 2. BLOCK if opponent can win
            opp = -player
            for m in legal:
                nb = apply_move(b, m, opp)
                if winner(nb) == opp:
                    return m

            return None


        def rollout(b, player):
            current = player
            bb = b.copy()

            while True:
                w = winner(bb)
                if w != 0:
                    return w

                legal = get_legal(bb)
                mv = legal[np.random.randint(len(legal))]
                bb = apply_move(bb, mv, current)
                current = -current

        def select_ucb(moves, wins, plays):
            total = sum(plays[m] for m in moves) + 1e-9
            best_score = -1
            best_move = moves[0]

            for m in moves:
                if plays[m] == 0:
                    return m

                exploit = wins[m] / plays[m]
                explore = C * np.sqrt(np.log(total) / plays[m])
                score = exploit + explore

                if score > best_score:
                    best_score = score
                    best_move = m

            return best_move

        def mcts(root_board, root_player, legal_moves):
            wins = {m: 0 for m in legal_moves}
            plays = {m: 0 for m in legal_moves}

            for _ in range(ITER):
                move = select_ucb(legal_moves, wins, plays)
                nextb = apply_move(root_board, move, root_player)
                reward = rollout(nextb, -root_player)

                plays[move] += 1
                if reward == root_player:
                    wins[move] += 1
                elif reward == 2:
                    wins[move] += 0.5

            # robust child = acción más visitada
            return max(legal_moves, key=lambda m: plays[m])

        
        #   EJECUCIÓN PRINCIPAL

        player = infer_player(board)
        legal = get_legal(board)

        # If only one legal move → play it
        if len(legal) == 1:
            return legal[0]

        #FIRST: WIN OR BLOCK
        tact = immediate_tactics(board, player, legal)
        if tact is not None:
            return tact

        # Otherwise MCTS
        return mcts(board, player, legal)