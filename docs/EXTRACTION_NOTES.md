# CNMP から CNP への切り分け

## 結論

CNP 部分だけを独立させることは可能です。上流リポジトリの PyTorch 実装では、`CNMP` というクラス名になっているニューラルネットワーク自体が、ほぼそのまま deterministic CNP です。CNMP の `M` は **Movement** であり、削除可能な独立した「M モジュール」や mixture head があるわけではありません。

確認した上流スナップショット:

- Repository: <https://github.com/myunusseker/CNMP>
- Commit: `95045db72e9f0233e7029993c1c668ff00dc2eb8`
- Main implementation: `CNMP_PyTorch.ipynb`

## 残した CNP コア

1. 各 context pair `(x_i, y_i)` を共有 Encoder で表現 `r_i` にする。
2. `mean_i(r_i)` で順序不変な set representation `r` を作る。
3. 各 target `x_q` と `r` を連結し、Decoder で Gaussian の `mean` と `scale` を出す。
4. Gaussian negative log likelihood で Encoder と Decoder を同時学習する。
5. 可変数 context と複数 target の一括 query を扱う。

この実装では、さらに通常の研究コードとして使いやすいように batched input、padding mask、shape/device/dtype 検証を追加しました。

## 外した CNMP 固有部分

| CNMP 側の要素 | CNP 単体から外した理由 |
|---|---|
| `x = time (+ task parameters)` | CNP の `x` は任意の入力でよい |
| `y = trajectory / sensorimotor state` | CNP の `y` は任意の出力でよい |
| 同一 demonstration 内の軌道 sampling | 汎用 set sampling に置換 |
| 時刻の `[0, 1]` 正規化 | movement data の前処理だから |
| 全時刻 query と trajectory plot | movement 用の可視化だから |
| goal/current sensor を observation にする規約 | ロボット実行プロトコルだから |
| feedback、clock hold、PD controller | 論文にはあるが上流 notebook にも未実装 |
| NumPy の `.npy` データと IPython UI | CNP モデル本体には不要 |

## 上流 notebook との対応

| 上流 notebook | このフォルダー |
|---|---|
| `class CNMP` | `src/cnp/model.py` の `ConditionalNeuralProcess` |
| `log_prob_loss` | `src/cnp/losses.py` の `gaussian_nll` |
| `get_train_sample` | `src/cnp/sampling.py` の汎用 sampler |
| trajectory generator / plot / validation loop | 抽出対象外 |

上流実装をそのままコピーすると、`d_x`/`d_y`/`model`/データの global 依存、unbatched input 限定、empty context で NaN、GPU tensor への `.numpy()`、不安定な NumPy scale 変換などが残ります。そのため、論文の CNP 構造と上流の挙動を保った小さな書き直しにしています。

## 判断した曖昧点

- 論文の背景節は context 数を `1..n_max`、CNMP 節は `0..n_max` としています。しかし mean aggregation の empty set は未定義で、上流コードも必ず 1 点以上を sample します。この実装は silent NaN を避け、0 context を明示的に拒否します。
- 論文の式は scale に `softmax` と書いていますが、上流 PyTorch コードは `softplus` を使います。各出力次元の正の標準偏差を作る用途にも `softplus` が整合するため、上流コードに合わせました。
- 出力は target ごとの diagonal Gaussian です。target 間の共分散を持つ joint trajectory distribution や latent Neural Process ではありません。context が固定されると予測パラメータも決定的です。
- mean aggregation は context の順序に不変ですが、大きい context set の情報を固定長ベクトルへ圧縮する制約があります。

## ライセンス上の注意

確認時点で上流 repository に `LICENSE`/`COPYING` はありません。このフォルダーには上流 notebook、画像、`.npy` データをコピーせず、アルゴリズムと論文記述に基づく独立した最小実装だけを置いています。外部配布時は、引用に加えて上流著者の利用条件を確認してください。
