import random
import math

class State:
    def __init__(self, pieces = None, enemy_pieces = None):
        self.pieces = pieces if pieces is not None else [0] * 9
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [0] * 9

    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    # 負けかどうか
    def is_lose(self):
        # 3並びかどうか
        def is_comp(x, y, dx, dy):
            for k in range(3):
                if y < 0 or 2 < y or x < 0 or 2 < x or \
                    self.enemy_pieces[x+y * 3] == 0:
                    return False
                x, y = x + dx, y + dy
            return True

        # 負けかどうか
        if is_comp (0, 0, 1, 1) or is_comp(0, 2, 1, -1):
            return True
        for i in range(3):
            if is_comp(0, i, 1, 0) or is_comp(i, 0, 0, 1):
                return True
        return False

    # 引き分けかどうか
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 9

    # ゲーム終了かどうか
    def is_done(self):
        return self.is_lose() or self.is_draw()

    # 次の状態の取得
    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    # 合法手のリストの取得
    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    # 先手かどうか
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # 文字列表示
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 3 == 2:
                str += '\n'
        return str

# ランダムで行動選択
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]

# ミニマックス法で状態価値計算
def mini_max(state):
    if state.is_lose():
        return -1
    elif state.is_draw():
        return 0

    # 合法手の状態価値の計算
    best_score = -float('inf')              # 初期値は負の無限大
    for action in state.legal_actions():    # 合法手を全て試す
        next_state = state.next(action)     # 次の状態を取得
        score = -mini_max(next_state)       # ミニマックス法で次の状態の価値を計算
        if score > best_score:              # スコアが最良なら
            best_score = score              # 最良スコアを更新

    return best_score                       # 最良スコアを返す

# ミニマックス法で行動選択
def mini_max_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    best_score = -float('inf')  # 初期値は負の無限大
    str = ['', '']
    for action in state.legal_actions():  # 合法手を全て試す
        score = -mini_max(state.next(action))  # ミニマックス法で次の状態の価値を計算
        if score > best_score:
            best_action = action
            best_score = score

        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('actions:', str[0], '\nscore:', str[1], '\n')

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action

def alpha_beta(state, alpha, beta):
    if state.is_lose():
        return -1
    elif state.is_draw():
        return 0

    # 合法手の状態価値の計算
    for action in state.legal_actions():  # 合法手を全て試す
        next_state = state.next(action)   # 次の状態を取得
        score = -alpha_beta(next_state, -beta, -alpha)  # α-β法で次の状態の価値を計算
        if score > alpha:  # α更新
            alpha = score

        if score >= beta:  # βカット
            return score

    return alpha  # 合法手の最大値αを返す

def alpha_beta_action(state):
    # 合法手の状態価値の計算
    best_action = 0
    best_score = -float('inf')  # 初期値は負の無限大
    str = ['', '']
    for action in state.legal_actions():  # 合法手を全て試す
        score = -alpha_beta(state.next(action), -float('inf'), float('inf'))  # α-β法で次の状態の価値を計算
        if score > best_score:
            best_action = action
            best_score = score

        str[0] = '{}{:2d},'.format(str[0], action)
        str[1] = '{}{:2d},'.format(str[1], score)
    print('actions:', str[0], '\nscore:', str[1], '\n')

    # 合法手の状態価値の最大値を持つ行動を返す
    return best_action

def playout(state):
    # 負けは状態価値： -1、引き分けは0
    if state.is_lose():
        return -1
    elif state.is_draw():
        return 0

    # 次の状態の状態価値
    return -playout(state.next(random_action(state)))

def argmax(collection, key = None):
    return collection.index(max(collection))

# モンテカルロ木探索の行動選択
def mcts_action(state):
    # モンテカルロ木探索のノードの定義
    class Node:
        def __init__(self, state):
            self.state = state  # 状態
            self.w = 0          # 累計価値
            self.n = 0          # 試行回数
            self.child_nodes = []  # 子ノードのリスト

        # 局面の価値の計算
        def evaluate(self):
            # ゲーム終了時
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0 # 負けなら-1、引き分けなら0

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

            # 子ノードが存在しない場合
            if not self.child_nodes:
                value = playout(self.state)

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1

                # 子ノードの展開
                if self.n == 10:
                    self.expand()
                return value

            # 子ノードが存在する場合
            else:
                value = self.next_child_node().evaluate()

                # 累計価値と試行回数の更新
                self.w += value
                self.n += 1
                return value

        # 子ノードの展開
        def expand(self):
            legal_actions = state.legal_actions() # 合法手の取得
            self.child_nodes = []
            for action in legal_actions:
                self.child_nodes.append(Node(self.state.next(action)))

        # UBC1が最大の子ノードを選択
        def next_child_node(self):
            # 試行回数が0の子ノードを返す
            for child_node in self.child_nodes:
                if child_node.n == 0:
                    return child_node

            # UBC1の計算
            t = 0
            for c in self.child_nodes:
                t += c.n
            ucb1_values = []
            for child_node in self.child_nodes:
                ucb1_values.append(-child_node.w / child_node.n + (2 * math.log(t) / child_node.n)**0.5)

            # UBC1が最大の子ノードを返す
            return self.child_nodes[argmax(ucb1_values)]


    # 現在の局面のノードを作成
    root_node = Node(state)
    root_node.expand()

    # 100回のシミュレーションを行う
    for _ in range(100):
        root_node.evaluate()

    # 試行回数の最大値を持つ行動を返す
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[argmax(n_list)]

# 動作確認
if __name__ == '__main__':
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 次の状態の取得
        state = state.next(random_action(state))

        # 状態の表示
        print(state)
        print()
