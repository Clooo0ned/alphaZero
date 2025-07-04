from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle
import os

# パラメータ
SP_GAME_COUNT =20 # 自己対戦のゲーム数
SP_TEMPERATURE = 0.1 # 温度パラメータ

# 先手プレイヤーの価値
def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = f'./data/{now.strftime("%Y%m%d_%H%M%S")}.history'
    with open(path, 'wb') as f:
        pickle.dump(history, f)

# 1ゲームの実行
def play(model):
    # 学習データ
    history = []

    # 状態の生成
    state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得
        scores = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE # 方策配列の初期化
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value  # 次のループではターンが変わりプレイヤーが交代するため、価値を反転する。
    return history

# 自己対戦の実行
def self_play():
    # 学習データ
    history = []

    # ベストプレイヤーの読み込み
    model = load_model('./model/best.h5')

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        h = play(model)
        history.extend(h)

        # 出力
        print(f'\rSelf Play {i+1}/{SP_GAME_COUNT}', end='')
    print('')

    # 学習データの保存
    write_data(history)

    # モデルの破棄
    K.clear_session()
    del model

# 動作確認
if __name__ == '__main__':
    self_play()
    print('Self Play completed.')
    print('Data saved in ./data/')
    print('You can use the data for training.')