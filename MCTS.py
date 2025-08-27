import math
import random
import numpy as np

class MCTSNode():
    def __init__(self, game, parent=None, prior_value=0, c_puct=0.8):
        self.game = game
        self.parent = parent
        self.c_puct = c_puct
        self.prior_value = prior_value
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0

    def is_terminal(self):
        return self.game.check_game_state() != 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self):
        best_score = float('-inf')
        best_move = None

        for move, (child, prior) in self.children.items():
            if child is None:
                score = self.c_puct * prior * np.sqrt(self.visit_count)
            else:
                score = child.value() + self.c_puct * prior * np.sqrt(self.visit_count) / (child.visit_count + 1)
            
            if score > best_score:
                best_score = score
                best_move = move

        best_child, prior = self.children[best_move]
        if best_child is None:
            x, y = best_move
            next_game = self.game.game_after_move(x, y)
            best_child = MCTSNode(next_game, self, prior)
            self.children[best_move] = best_child, prior
        
        return best_child
    
    def backpropagate(self, value):
        node = self

        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def expand(self, model):
        board_tensor = self.game.board2tensor()
        valid_moves = self.game.get_valid_moves()
        self.prior_value, policy = model.predict(board_tensor)

        valid_moves_policy = sum(policy[x][y] for x, y in valid_moves)
        if valid_moves_policy == 0:
            valid_moves_policy += 1e-10

        dir_noise = np.random.dirichlet([0.03] * len(valid_moves))
        for (x, y), noise in zip(valid_moves, dir_noise):
            p = 0.75 * policy[x][y] / valid_moves_policy + 0.25 * noise
            self.children[(x, y)] = None, p

class MCTS():
    def __init__(self, game, model, c_puct=0.8, simulations_num=800):
        self.game = game
        self.model = model
        self.c_puct = c_puct
        self.simulations_num = simulations_num
        self.root = MCTSNode(game=game, parent=None, c_puct=c_puct)

    def search(self):
        for _ in range(self.simulations_num):
            node = self.root

            while len(node.children) > 0:
                node = node.select_child()

            if node.is_terminal():
                val = node.game.get_terminal_val()
            else:
                node.expand(self.model)
                val = node.prior_value

            node.backpropagate(val)

        return self.get_moves_probs()
    
    def get_moves_probs(self, temperature=1.0):
        moves_visits = {}
        for move, (child, prior) in self.root.children.items():
            moves_visits[move] = child.visit_count if child is not None else 0

        if temperature == 0:
            best_move = max(moves_visits, key=moves_visits.get)
            probs = {move: 0 for move in moves_visits}
            probs[best_move] = 1.0
        else:
            actions = list(moves_visits.keys())
            visits = np.array(list(moves_visits.values()), dtype=np.float32)
            visits = visits ** (1 / temperature)
            visits /= np.sum(visits)

            probs = {a: v for a, v in zip(actions, visits)}

        return probs