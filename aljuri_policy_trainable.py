import numpy as np
import hashlib
import json
from connect4.policy import Policy


class ALjuriRuiz(Policy):

    def __init__(self, gamma=0.99, q_filename=None):
        self.Q = {}
        self.N = {}
        self.episode = []
        self.gamma = gamma

        # Cargar memoria si se pasa un JSON
        if q_filename is not None:
            try:
                with open(q_filename, "r") as f:
                    data = json.load(f)
                for key, val in data["Q"].items():
                    s, a = key.split("|")
                    self.Q[(s, int(a))] = val
                for key, val in data["N"].items():
                    s, a = key.split("|")
                    self.N[(s, int(a))] = val
                print("[INFO] Memoria cargada con", len(self.Q), "entradas")
            except:
                print("[WARNING] No se encontró el JSON. Empezando desde cero.")

    # ------------------------------
    # Utils
    # ------------------------------

    def encode(self, board):
        return hashlib.sha1(board.tobytes()).hexdigest()

    def save(self, path="Q_table.json"):
        data = {
            "Q": {f"{k[0]}|{k[1]}": v for k, v in self.Q.items()},
            "N": {f"{k[0]}|{k[1]}": v for k, v in self.N.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="Q_table.json"):
        with open(path, "r") as f:
            data = json.load(f)
        self.Q.clear()
        self.N.clear()
        for key, val in data["Q"].items():
            s, a = key.split("|")
            self.Q[(s, int(a))] = val
        for key, val in data["N"].items():
            s, a = key.split("|")
            self.N[(s, int(a))] = val

    def mount(self, time_out=None):
        self.episode = []

    # ------------------------------
    # Immediate Tactics (WIN - BLOCK)
    # ------------------------------

    def immediate_tactics(self, b, player, legal):

        def apply_move(board, col, p):
            newb = board.copy()
            for r in range(5, -1, -1):
                if newb[r, col] == 0:
                    newb[r, col] = p
                    break
            return newb

        def winner(bb):
            # Horizontal
            for r in range(6):
                for c in range(4):
                    line = bb[r, c:c+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1

            # Vertical
            for c in range(7):
                col = bb[:, c]
                for r in range(3):
                    line = col[r:r+4]
                    if np.all(line == 1): return 1
                    if np.all(line == -1): return -1

            # Diagonal derecha
            for r in range(3):
                for c in range(4):
                    diag = np.array([bb[r+i, c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            # Diagonal izquierda
            for r in range(3):
                for c in range(3,7):
                    diag = np.array([bb[r+i, c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            return 0

        # 1. WIN MOVE
        for m in legal:
            if winner(apply_move(b, m, player)) == player:
                return m

        # 2. BLOCK OPPONENT WIN
        opp = -player
        for m in legal:
            if winner(apply_move(b, m, opp)) == opp:
                return m

        return None

    # ------------------------------
    # ACT = Tactics + Q + MCTS
    # ------------------------------

    def act(self, board: np.ndarray) -> int:

        reds = np.sum(board == -1)
        yellows = np.sum(board == 1)
        player = -1 if reds == yellows else 1

        legal = [c for c in range(7) if board[0, c] == 0]
        state = self.encode(board)

        # 1. Immediate tactics (win/block)
        tact = self.immediate_tactics(board, player, legal)
        if tact is not None:
            self.episode.append((state, tact))
            return tact

        # 2. Q-table if known
        known = [(self.Q[(state, a)], a) for a in legal if (state, a) in self.Q]
        if known:
            _, action = max(known)
            self.episode.append((state, action))
            return action

        # 3. MCTS fallback (new unknown states)
        action = self.mcts(board)
        self.episode.append((state, action))
        return action

    # ------------------------------
    # MCTS EXACTO (VERSIÓN BUENA)
    # ------------------------------

    def mcts(self, board):
        ITER = 10
        C = 1.41

        def infer_player(b):
            reds = np.sum(b == -1)
            yell = np.sum(b == 1)
            return -1 if reds == yell else 1

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

            # Diag derecha
            for r in range(3):
                for c in range(4):
                    diag = np.array([b[r+i, c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            # Diag izquierda
            for r in range(3):
                for c in range(3,7):
                    diag = np.array([b[r+i, c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            if np.all(b[0] != 0):
                return 2
            return 0

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
            best = moves[0]
            best_score = -1
            for m in moves:
                if plays[m] == 0:
                    return m
                exploit = wins[m] / plays[m]
                explore = C * np.sqrt(np.log(total) / plays[m])
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best = m
            return best

        player = infer_player(board)
        legal = get_legal(board)

        wins = {m: 0 for m in legal}
        plays = {m: 0 for m in legal}

        for _ in range(ITER):
            m = select_ucb(legal, wins, plays)
            b2 = apply_move(board, m, player)
            result = rollout(b2, -player)
            plays[m] += 1
            if result == player:
                wins[m] += 1
            elif result == 2:
                wins[m] += 0.5

        return max(legal, key=lambda m: plays[m])

    # ------------------------------
    # FVMC LEARNING
    # ------------------------------

    def learn(self, reward):
        G = reward
        visited = set()

        for (s, a) in reversed(self.episode):
            if (s, a) not in visited:
                visited.add((s, a))

                if (s, a) not in self.N:
                    self.N[(s, a)] = 0
                    self.Q[(s, a)] = 0.0

                self.N[(s, a)] += 1
                self.Q[(s, a)] += (G - self.Q[(s, a)]) / self.N[(s, a)]

            G *= self.gamma

        self.episode = []
