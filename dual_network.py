from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

# パラメータの準備
DN_FILTERS = 128            # 畳み込み層のカーネル数
DN_RESIDUAL_NUM = 16        # 残差ブロックの数
DN_INPUT_SHAPE = (3, 3, 2)  # 入力の形状
DN_OUTPUT_SHAPE = 9         # 行動数(配置先(3*3))

# 畳み込み層の定義
def conv(filters):
    return Conv2D(filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

# 残差ブロックの定義
def residual_block():
    def f(x):
        sc = x                      # スキップ接続のための入力を保存
        x = conv(DN_FILTERS)(x)     # 畳み込み層
        x = BatchNormalization()(x) # 正規化
        x = Activation('relu')(x)   # 活性化関数
        x = conv(DN_FILTERS)(x)     # 畳み込み層
        x = BatchNormalization()(x) # 正規化
        x = Add()([x, sc])          # スキップ接続
        x = Activation('relu')(x)   # 活性化関数
        return x
    return f

# デュアルネットワークのモデルを定義
def dual_network():
    # モデル作成済みの場合は無処理
    if os.path.exists('./model/best.h5'):
        return

    # 入力層
    inputs = Input(DN_INPUT_SHAPE)

    # 畳み込み層
    x = conv(DN_FILTERS)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 残差ブロック*16
    for _ in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)

    # プーリング層
    x = GlobalAveragePooling2D()(x)

    # ポリシー出力
    p = Dense(DN_OUTPUT_SHAPE, activation='softmax', kernel_regularizer=l2(0.0005), name='pi')(x)

    # バリュー出力
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh', name='v')(v)

    # モデルの作成
    model = Model(inputs=inputs, outputs=[p, v])

    # モデルの保存
    os.makedirs('./model/', exist_ok=True)  # モデル保存用ディレクトリの作成
    model.save('./model/best.h5')                 # ベストプレイヤーのモデルを保存

    # モデルの破棄
    K.clear_session()
    del model

# 動作確認
if __name__ == '__main__':
    dual_network()  # デュアルネットワークのモデルを作成
    print("Dual network model created and saved.")