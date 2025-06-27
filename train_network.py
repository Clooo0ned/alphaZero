import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

from dual_network import DN_INPUT_SHAPE
from tensorflow.keras.callbacks import LearningRateScheduler, LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
import pickle

# パラメータ
RN_EPOCHS = 100 # 学習エポック数

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data/').glob('*.history'))[-1]  # 最新の学習データファイルを取得
    with history_path.open(mode = 'rb') as f:
        return pickle.load(f)

# デュアルネットワークの学習
def train_network():
    # 学習データの読み込み
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # 入力データの整形
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)  # (N, a, b, c) の形状に変換
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # モデルの読み込み
    model = load_model('./model/best.h5')

    # モデルのコンパイル
    model.compile(loss = ['categorical_crossentropy', 'mse'], optimizer = 'adam')

    # 学習率
    def step_decay(epoch):
        x = 0.001
        if epoch >= 50: x = 0.0005
        if epoch >= 80: x = 0.00025
        return x
    lr_decay = LearningRateScheduler(step_decay)

    # 出力
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print(f'\rTrain {epoch + 1}/{RN_EPOCHS}', end=''))

    # 学習の実行
    model.fit(xs, [y_policies, y_values], epochs=RN_EPOCHS, batch_size=128, verbose=0,
              callbacks=[lr_decay, print_callback])
    print("")

    # モデルの保存
    model.save('./model/latest.h5')

    # モデルの破棄
    K.clear_session()
    del model

# 動作確認
if __name__ == '__main__':
    train_network()
    print("Network training completed.")