import numpy as np
import copy

class GomokuGame:
    def __init__(self, size=15):
        self.board_size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1

    def clear(self):
        self.board[:] = 0
        self.current_player = 1

    def get_board_size(self):
        return self.board_size

    def get_valid_moves(self):
        moves = []
        for i, row in enumerate(self.board):
            for j, block in enumerate(row):
                if block == 0:
                    moves.append((i, j))
        return moves
    
    def is_move_out_of_bounds(self, x, y):
        return x < 0 or x >= self.board_size or y < 0 or y >= self.board_size

    def check_game_state(self):
        '''
        查询游戏状态
        返回值:
            0 游戏未结束
            1 先手玩家获胜
           -1 后手玩家获胜
            2 平局
        '''
        space = self.board_size * self.board_size
        for i, row in enumerate(self.board):
            for j, block in enumerate(row):
                if block != 0:
                    space -= 1
                    for x, y in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                        cnt = 1
                        for k in range(1, 5):
                            ni = i + x * k
                            nj = j + y * k
                            if self.is_move_out_of_bounds(ni, nj) or self.board[i][j] != self.board[ni][nj]:
                                break
                            else:
                                cnt += 1
                            
                            if cnt == 5:
                                return self.board[i][j]
        if space > 0:
            return 0
        else:
            return 2

    def get_game_state(self):
        '''
        返回 board, current_player
        board: 当前棋盘
        current_player: 当前玩家
        '''
        return self.board.copy(), int(self.current_player)

    def board2tensor(self):
        '''
        将棋盘转为3通道tensor
        分别为 当前方棋子 对方棋子 空位
        '''
        cur_player = (self.board == self.current_player).astype(np.float32)
        opponent_player = (self.board == -self.current_player).astype(np.float32)
        empty = (self.board == 0).astype(np.float32)

        tensor = np.stack([cur_player, opponent_player, empty], axis=0)
        return tensor

    def move(self, x, y):
        if self.is_move_out_of_bounds(x, y) or self.check_game_state() != 0 or self.board[x][y] != 0:
            return False
        self.board[x][y] = self.current_player
        self.current_player = -self.current_player
        return True
    
    def game_after_move(self, x, y):
        # next_game = copy.deepcopy(self)
        next_game = GomokuGame(self.get_board_size())
        next_game.current_player = self.current_player
        next_game.board = self.board.copy()
        if not next_game.move(x, y):
            return None
        return next_game

    def get_terminal_val(self):
        result = self.check_game_state()
        if result == 0 or result == 2:
            return 0

        used = np.count_nonzero(self.board)
        return result * (1 - used * 1e-3)

    def display(self):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print('  ' + ' '.join(str(i) for i in range(self.board_size)))
        for r in range(self.board_size):
            print(f'{r} ' + ' '.join(symbols[self.board[r][c]] for c in range(self.board_size)))
        print()