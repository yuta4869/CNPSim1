# Sin/Cos synthetic CNP benchmark

## 結論

基本CNPは、未知のSin/Cos関数に対する通常の補間を学習できました。未知ID関数の全域補間RMSEは、Context 3点の`0.5296`から20点の`0.1829`まで低下しています。したがって、CNP実装、Context/Queryの入力、Gaussian NLLによる学習パイプラインがまったく機能していないわけではありません。

一方、心拍の連続欠損に近い中央Gap補間はContext 5点の`0.1798`が最良で、20点では`0.2881`へ悪化しました。全域からランダムにContextを取る学習と、全Contextを単純平均する基本CNPが、Gap両側の局所情報を使う問題に合っていないことを示します。これはBIDMCで線形補間に敗けた結果と整合します。

このrunは実装診断用の1 seed実験です。Sin/Cosで成功したことは、心拍への有効性や臨床性能を意味しません。

## なぜSinとCosを使うのか

1本のfunction taskを次で生成します。

\[
y(x)=b+A\sin(\omega x+\phi)
\]

- 振幅: \(A\in[0.5,2.0]\)
- 角周波数: \(\omega\in[0.8,2.0]\)
- 位相: \(\phi\in[0,2\pi)\)
- オフセット: \(b\in[-0.5,0.5]\)
- 正規化時刻: \(x\in[-2,2]\)
- 観測ノイズ: なし

\(\cos(\theta)=\sin(\theta+\pi/2)\)なので、位相を変えるこの関数族はSinとCosの両方を含みます。モデルへ振幅、周波数、位相、Sin/Cosラベルは渡しません。CNPが受け取るのはContextの`(x, y)`とQueryの`x`だけです。pure Sin testは`phase=0, offset=0`、pure Cos testは`phase=pi/2, offset=0`へ固定しています。

Sin/Cosは心拍そのものを模擬するためではなく、次を切り分ける既知の関数分布です。

1. CNPが複数のfunctionから共通規則を学べるか
2. 未知functionでもContextから振幅・周波数・位相を推測できるか
3. Query時刻だけを与えてYを返す入出力が正しく実装されているか
4. Context増加に平均誤差と予測区間が反応するか

## Function bankとsplit

パラメータを先に固定し、完全一致するtupleがsplit間にないことを検査しています。

| Split | 関数数 | 用途 |
|---|---:|---|
| Train | 4,096 | 学習。Context/Query位置を毎episode再sample |
| Validation | 256 | checkpoint選択だけに使用 |
| ID test | 256 | trainと同じparameter範囲の未知関数 |
| Frequency OOD test | 256 | \(\omega\in[2.2,3.0]\)の未知関数 |
| Pure Sin test | 128 | \(\phi=0,b=0\) |
| Pure Cos test | 128 | \(\phi=\pi/2,b=0\) |

Parameter bank、学習sampling、validation、testには別seedを使います。bankのseedとSHA-256 fingerprint、validation/testのContext seedは`results.json`へ保存されています。testはbest checkpointをvalidationだけで選択してreloadした後に1回だけ評価します。

## 1回の学習で何を入れたか

各optimizer stepでは次を行います。

1. Train bankから64関数を一様に選ぶ
2. 各関数のContext数を3〜20から独立に選ぶ
3. Context時刻と32個のQuery時刻を`[-2,2]`から独立にsampleする
4. Contextの`(x, y)`とQueryの`x`をCNPへ入力する
5. 隠したQueryの`y`をGaussian NLLの正解としてだけ使う
6. 64関数×32 Queryのlossを平均してAdamで1回更新する

```text
Context (x, y) ---- Encoder ---- mean aggregation ---- r
                                                       |
Query x ------------------------------------------------+---- Decoder ---- mean, scale
Hidden Query y ----------------------------------------------------------- Gaussian NLL
```

batch内でContext数が異なるため、最大20点へpaddingし`context_mask`で無効点を除外します。Hidden Query Yはモデル入力に含まれません。

モデルと最適化:

- deterministic CNP、66,818 trainable parameters
- hidden 128、representation 128
- Adam、learning rate `1e-4`
- Gaussian NLL
- gradient global norm clipping `1.0`
- validation every 250 steps
- validationの2補間条件×4 Context数の平均NLLでcheckpoint選択
- 最大10,000 steps、minimum 2,500、patience 12

採用checkpointは上限のstep 10,000でした。

| 学習量 | 値 |
|---|---:|
| Optimizer updates | 10,000 |
| Function episodes | 640,000 |
| 使用した固有train functions | 4,096 / 4,096 |
| Context observations | 7,362,623 |
| Target loss points | 20,480,000 |
| Best validation NLL | 0.2972 |
| Best validation RMSE | 0.2891 |

bestが最大stepにあるため、完全収束したとは断定できません。今回の目的には十分な改善が見えましたが、正式比較では上限延長と複数seedが必要です。

## 固定評価

Context候補とQueryをinterleaved gridの別座標に置き、Context 3/5/10/20のすべてで同じQuery地点を評価します。少ないContext集合は多い集合のsubsetです。

- `interpolation`: `x=-2,2`を必ずContextにして、全QueryをContext hull内に置く
- `gap_interpolation`: Contextを`|x|>=0.6`、Queryを中央Gap `|x|<0.6`に置く
- `extrapolation_stress`: Contextを`[-1,1]`、Queryをその外側に置く

最後の条件はContext配置も学習時と異なるstress testです。checkpoint選択には使わず、心拍の未来予測を学習したという意味でもありません。

評価値はfunctionごとに計算してからmacro平均します。CNPはMAE、RMSE、Gaussian NLL、68/95% coverage、95%区間幅を報告します。予測区間はQueryごとのpointwise区間で、関数全体の同時区間ではありません。

比較対象:

- Context平均
- 最近傍
- 区分線形補間。Context hull外では端2点による線形外挿
- Sin/Cos regression: 周波数gridを探索し、`b + alpha*sin(wx) + beta*cos(wx)`をfitするfamily-aware baseline

Sin/Cos regressionは正しい関数族を事前に知る強い上限目安です。周波数が同定できないContext 3点では報告しません。

## Test結果

### 未知ID関数のCNP RMSE

| 条件 | C=3 | C=5 | C=10 | C=20 |
|---|---:|---:|---:|---:|
| 全域補間 | 0.5296 | 0.3615 | 0.2369 | **0.1829** |
| 中央Gap補間 | 0.1818 | **0.1798** | 0.2381 | 0.2881 |
| 外挿stress | 0.4983 | 0.4779 | **0.4668** | 0.4868 |

全域補間ではContext増加に従って単調に改善しました。中央Gapでは逆に10点以降で悪化し、予測区間幅も`0.8078`から`1.2836`へ広がりました。追加Contextを局所的に選別できず、平均表現がContext配置の変化を受ける基本CNPの弱点が現れています。

### ID関数、Context 20のbaseline比較

| 条件 | CNP | Linear | Nearest | Context mean | Sin/Cos fit |
|---|---:|---:|---:|---:|---:|
| 全域補間 | 0.1829 | 0.0405 | 0.1439 | 0.8233 | **0.0031** |
| 中央Gap補間 | 0.2881 | 0.2022 | 0.4148 | 0.8782 | **0.0027** |
| 外挿stress | 0.4868 | 0.3877 | 0.6142 | 1.1284 | **0.0057** |

CNPは全域補間のContext 3/5では線形より良いものの、Context 10/20では線形が上回ります。中央GapはContext 3/5でCNPが線形よりわずかに良く、20では敗けます。正しいSin/Cos familyを明示した回帰はほぼ完全に復元します。

### Pure Sin/Cosと周波数OOD

| Test / 条件 | C=3 | C=5 | C=10 | C=20 |
|---|---:|---:|---:|---:|
| Pure Sin / 全域補間 | 0.6081 | 0.4038 | 0.2603 | 0.2044 |
| Pure Cos / 全域補間 | 0.3646 | 0.3016 | 0.2128 | 0.1559 |
| Frequency OOD / 全域補間 | 1.2065 | 1.0595 | 0.8837 | 0.7662 |
| Pure Sin / 中央Gap | 0.1759 | 0.1580 | 0.1926 | 0.2087 |
| Pure Cos / 中央Gap | 0.2074 | 0.2063 | 0.2339 | 0.2744 |
| Frequency OOD / 中央Gap | 0.6941 | 0.8068 | 0.9573 | 1.0437 |

pure Sin/Cosの両方を予測できていますが、同等精度ではありません。未知の高周波では大幅に悪化するため、学習range外へ一般化できたとはいえません。

### 不確実性

ID全域補間ではContext 3→20で95%区間幅が`2.0390→0.9707`へ縮み、coverageは`89.68%→98.16%`になりました。Contextから関数が絞られる方向は学習していますが、20点では区間がやや保守的です。

中央GapではContext増加に対して誤差と区間幅がともに増え、68% coverageも理想値付近で安定しません。このGaussian headはtarget間相関を表現せず、校正済みの関数不確実性とはみなせません。

## 何が分かったか

1. `Context (X,Y) + Query X -> Prediction Y`のCNP経路は学習できている。
2. Sin/Cosは別々のnetworkではなく、位相を含む1つのfunction distributionとして学習できる。
3. 基本CNPは通常のランダム補間にはContext増加を利用できる。
4. 構造化GapではContext増加が自動的な改善を保証しない。
5. 周波数range外、pure Sin/Cos間の精度差、不確実性校正は未解決。
6. BIDMCの失敗を「学習step不足」だけで説明する根拠は弱く、Contextの局所性とtraining mask分布が重要。

次の優先実験は、同じ固定bankを使った次の比較です。

1. random episodeに中央Gap episodeを混ぜる
2. basic mean CNPとAttentive CNP / local-context CNPを比較する
3. seed `7,17,27`で再実行し、function単位bootstrap CIを加える
4. 10,000 step時点で改善中なので、20,000 step上限も比較する
5. その後にBIDMCへ同じ構造変更を移す

## 再実行

```bash
python examples/train_sincos.py --device auto
```

収録runの環境:

- Python 3.10.20
- PyTorch 2.13.0
- macOS 15.7.4 arm64
- Apple MPS
- seed 7

環境差、特にMPSにより数値の完全一致は保証しません。全数値は`outputs/sincos/results.json`を正とします。

## 成果物

- `outputs/sincos/best_model.pt`
- `outputs/sincos/results.json`
- `outputs/sincos/learning_curve.png`
- `outputs/sincos/context_sweep.png`
- `outputs/sincos/anchor_predictions.png`

SHA-256:

```text
7de3c1caaba058a0df4dfe734ea6ec40ecf5de56d3c8d113e513b1458cd7cb79  best_model.pt
98103d4778b5ba6748b0a7c2fcea31970b317c4c4e290870186c8d11ffa219b8  results.json
108b89ea196f432efdb0c8f0a55f7848535b055e92983d83aa8e6301308c04e6  learning_curve.png
75cab45353c92d8674e33738e8f5b7f9d7d6c7ff52a256318dd82115bedbdac5  context_sweep.png
bd1e8f968bacea48b1deb70efd85a564875a88db1818c6b09b6acbbf2eb06a1e  anchor_predictions.png
```
