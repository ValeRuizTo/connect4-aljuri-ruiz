import numpy as np
import hashlib
import json
from connect4.policy import Policy

class ALjuriRuiz(Policy):

    def __init__(self, gamma=0.99):
        # Memoria entre partidas
        self.Q = {}  # Q[(state_hash, action)]
        self.N = {}  # N[(state_hash, action)]
        self.episode = []
        self.gamma = gamma

    # ---------------------------------------
    # UTILIDADES
    # ---------------------------------------

    def encode(self, board):
        return hashlib.sha1(board.tobytes()).hexdigest()

    def save(self, path="Q_table.json"):
        """Guardar la memoria aprendida."""
        data = {
            "Q": {f"{k[0]}|{k[1]}": v for k, v in self.Q.items()},
            "N": {f"{k[0]}|{k[1]}": v for k, v in self.N.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path="Q_table.json"):
        """Cargar memoria previamente aprendida."""
        with open(path, "r") as f:
            data = json.load(f)
        self.Q = {}
        self.N = {}
        for key, val in data["Q"].items():
            s, a = key.split("|")
            self.Q[(s, int(a))] = val
        for key, val in data["N"].items():
            s, a = key.split("|")
            self.N[(s, int(a))] = val

    # ---------------------------------------
    # MOUNT (se llama al inicio de cada match)
    # ---------------------------------------

    def mount(self, time_out=None):
        # No limpiamos memoria entre partidas.
        self.episode = []

    # ---------------------------------------
    # ACT (MCTS + Q-learning)
    # ---------------------------------------

    def act(self, board: np.ndarray) -> int:

        # Detectar jugador actual
        reds = np.sum(board == -1)
        yellows = np.sum(board == 1)
        player = -1 if reds == yellows else 1

        # Acciones legales
        legal = [c for c in range(7) if board[0, c] == 0]

        state = self.encode(board)

        # Si ya aprendimos algo → greedy en Q
        known = []
        for a in legal:
            if (state, a) in self.Q:
                known.append((self.Q[(state, a)], a))

        if known:
            # Policy Improvement (mejor acción conocida)
            _, action = max(known)
        else:
            # Explorar con MCTS en estados desconocidos
            action = self.mcts(board)

        # Registrar para FVMC
        self.episode.append((state, action))

        return action

    # ---------------------------------------
    #  MCTS PARA ESTADOS NO APRENDIDOS
    # ---------------------------------------

    def mcts(self, board):
        ITER = 15
        C = 1.41

        def get_legal(b):
            return [c for c in range(7) if b[0, c] == 0]

        def apply_move(b, col, player):
            bb = b.copy()
            for r in range(5, -1, -1):
                if bb[r, col] == 0:
                    bb[r, col] = player
                    break
            return bb

        def infer(b):
            reds = np.sum(b == -1)
            yell = np.sum(b == 1)
            return -1 if reds == yell else 1

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
                    diag = np.array([b[r+i,c+i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1
            # Diag izquierda
            for r in range(3):
                for c in range(3,7):
                    diag = np.array([b[r+i,c-i] for i in range(4)])
                    if np.all(diag == 1): return 1
                    if np.all(diag == -1): return -1

            # Draw si fila superior llena
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
                if len(legal) == 0:
                    return 2  # empate

                mv = legal[np.random.randint(len(legal))]
                bb = apply_move(bb, mv, current)
                current = -current

        player = infer(board)
        legal = get_legal(board)

        if len(legal) == 0:
            return 0  # fallback (no debería ocurrir)

        wins = {a: 0 for a in legal}
        plays = {a: 0 for a in legal}

        for _ in range(ITER):
            for a in legal:
                b2 = apply_move(board, a, player)
                reward = rollout(b2, -player)
                plays[a] += 1
                if reward == player:
                    wins[a] += 1

        # Acción más visitada
        return max(legal, key=lambda a: plays[a])

    # ---------------------------------------
    #  LEARNING (FVMC)
    # ---------------------------------------

    def learn(self, reward):
        """Se llama cuando termina una partida, reward=+1/-1/0."""
        G = reward
        visited = set()

        for (s, a) in reversed(self.episode):
            if (s, a) not in visited:
                visited.add((s, a))
                if (s, a) not in self.N:
                    self.N[(s,a)] = 0
                    self.Q[(s,a)] = 0.0

                self.N[(s,a)] += 1
                self.Q[(s,a)] += (G - self.Q[(s,a)]) / self.N[(s,a)]

            G *= self.gamma

        self.episode = []
