# モンテカルロ木探索とランダムおよびアルファベータ法の比較
from game import State, mcts_action, random_action, alpha_beta_action

# パラメータ
EP_GAME_COUNT = 100  # 1評価あたりのゲーム数

# 先手プレイヤーのポイント
def first_player_point(ended_state):
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 1

# 1ゲームの実行
def play(next_actions):
    # 状態の生成
    state = State()

    # ゲーム終了までループ
    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 行動の選択
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 次の状態の取得
        state = state.next(action)

    # 先手プレイヤーのポイントを返す
    return first_player_point(state)

# 任意のアルゴリズムの評価
def evaluate_algorithm_of(label, next_actions):
    # 複数回の対戦を繰り返す
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # 1ゲームの実行
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))

        # 出力
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')

    # 平均ポイントを計算
    average_point = total_point / EP_GAME_COUNT
    print(label.format(average_point))

# VSランダム
next_actions = (mcts_action, random_action)
evaluate_algorithm_of('VS_Random: {:.3f}', next_actions)

# VSアルファベータ法
next_actions = (mcts_action, alpha_beta_action)
evaluate_algorithm_of('VS_AlphaBeta: {:.3f}', next_actions)