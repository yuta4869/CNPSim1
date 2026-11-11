# RBF-GP function benchmark

## 目的

固定された `sin(x)` / `cos(x)` は、CNPのforward・学習・Context条件付けが壊れていないことを確かめる最小テストです。しかし、2本を記憶するだけでも解けるため、「未知の関数を少数のContextから補間できる」ことまでは検証できません。

RBF-GP modeでは、各episodeでGaussian Process（GP）から新しい滑らかな関数を生成します。CNPは特定の1本を記憶するのではなく、滑らかな関数の**分布**と、Contextを使った条件付き予測を学ぶ必要があります。心拍数へ進む前に、CNPらしい性質が働くかを確認する中間benchmarkです。

この検証は、CNP原論文の最初の1次元関数回帰実験を参考にしています。原論文本文は「exponential kernel」と記載していますが、式やlength scaleは示していません。DeepMind公式notebookは「mean-squared exponential」と説明し、数学的にはsquared-exponential / RBFである式を実装しています。このrepositoryは、式が明示された公式notebook側に合わせた **squared-exponential / RBF kernelの固定parameter版**です。ただし、下記の通り学習量とsampling設計を意図的に変更しているため、原論文の数値を再現する実験ではありません。

- [Conditional Neural Processes（Garnelo et al., ICML 2018）](https://proceedings.mlr.press/v80/garnelo18a.html)
- [DeepMindの公式CNP notebook（参照した固定revision）](https://github.com/google-deepmind/neural-processes/blob/e7830a5664abf42c5eb891c48cf45423c4c21fc0/conditional_neural_process.ipynb)

## 生成する関数

0平均GPと、次のRBF kernelを使います。

```text
f ~ GP(0, k)
k(x, x') = sigma_f^2 exp(-(x - x')^2 / (2 ell^2))
```

既定値は次の通りです。

| 項目 | 既定値 | 意味 |
|---|---:|---|
| x range | `[-2, 2]` | 学習・評価するdomain |
| grid | 129点 | GP sampleを表現する座標 |
| length scale `ell` | `0.4` | 小さいほど細かく変動する |
| signal std `sigma_f` | `1.0` | 関数の典型的な振幅 |
| observation noise std | `0.02` | covariance対角へ加える独立noise。公式notebookと同値 |
| jitter | `1e-5` | Cholesky分解へさらに加える数値安定化項 |

有限grid上のRBF covarianceへ`observation_noise^2 + jitter`を加えて一度だけCholesky分解し、各episodeで独立な標準正規vectorを掛けて新しいnoisy GP realizationをsampleします。毎step同じbankを反復する方式ではありません。analytic GP posteriorも同じnoise仮定を使います。

## X、Y、Context、Query

1episodeは1本の未知関数です。

- `X`: 関数上の位置。将来の心拍数用途では正規化した時刻に相当
- `Y`: その位置の関数値。将来は心拍数に相当
- `Context`: CNPへ既知情報として渡す `(X, Y)` の組
- `Query`: `X`だけを渡して `Y`を予測させる位置
- `Prediction`: QueryごとのGaussian meanとscale

学習中は1episodeあたりContext数を3〜20からsampleし、Contextと重複しない32 Query点のGaussian negative log-likelihoodを最小化します。既定では64 episode/batch、最大10,000 optimizer stepなので、最後まで実行した場合は640,000本の独立なGP function episodeを生成します。

### 公式notebookとの差

公式notebookは、length scale `0.4`、signal scale `1.0`、covariance対角項 `(0.02)^2`、`x ∈ [-2, 2)`、batch 64、Adam learning rate `1e-4`で **200,000 iteration** 学習します。Context数は3〜9、追加target数は2〜9で、loss対象targetにはContext自身も含めます。

本modeはまず現実的な時間で検証できるよう最大10,000 step、Context 3〜20、32個のContext非重複Queryを使います。公式notebookと同じ`observation_noise=0.02`を使い、別途`jitter=1e-5`を数値安定化のために加えます。学習量とContext/Query samplingは異なるため、10,000 stepでgateを満たさない場合は、model破損と即断せず学習曲線を確認し、`--steps`と`--min-steps`を増やしてください。

## validation / testの分離

学習用関数はその場で新規生成します。評価だけは再現できるように固定bankを使います。

| split | 既定の関数数 | 用途 |
|---|---:|---|
| validation | 128 | best checkpointの選択とearly stopping |
| test | 256 | best checkpointをreloadした後の最終評価のみ |

validationとtestは独立したseedで生成し、各bankのSHA-256をcheckpointと`results.json`へ保存します。best checkpointは、Context 3 / 5 / 10 / 20におけるvalidationのfunction-macro RMSE平均が最小のstepです。test値をcheckpoint選択には使いません。

各Context条件では、Context位置をQueryから除外します。Context数を増やす場合は同じseedから作るnested Context setを使うため、Context数による変化を比較しやすくしています。

## 比較する3方式

| 方式 | 与えられる情報 | 役割 |
|---|---|---|
| CNP | 学習済みparameterと現在のContext | 検証対象 |
| Analytic GP posterior | Contextに加えて、真のRBF kernelとhyperparameter | kernelを知るoracle reference |
| Context mean | Context Yの平均だけ | 最小baseline |

GP posteriorは生成分布を正確に知っているため、CNPより有利です。原論文も、正しいhyperparameterを持つGPを上限相当の比較対象としています。このrepositoryでは、CNPがGP posteriorを上回ることを合格条件にはしません。

## 事前に固定した合格条件

held-out testを確認する前に、次の相対条件をcode上で固定しています。

1. CNPのContext 20 RMSEがContext 3より20%以上改善する。
2. CNPのContext 10 RMSEがContext meanより25%以上小さい。

1は追加Contextを利用できているか、2は単純な平均回帰以上の位置依存補間ができているかを見る条件です。絶対RMSE gateはkernelのlength scaleやdomain幅で難易度が変わるため、このmodeでは採用していません。

合格は「このsynthetic distributionで基本動作した」という意味だけです。心拍数への精度、人物差への頑健性、将来予測能力を保証しません。

## 実行

repository rootで実行します。

```bash
python examples/train_rbf_gp.py --device auto
```

短いsmoke testは次の通りです。これはpipeline確認用であり、合格判定用の学習量ではありません。

```bash
python examples/train_rbf_gp.py \
  --device cpu \
  --steps 2 \
  --min-steps 1 \
  --validation-interval 1 \
  --patience 2 \
  --batch-size 4 \
  --validation-functions 4 \
  --test-functions 4 \
  --no-plot \
  --output-dir /tmp/cnp-rbf-gp-smoke
```

主要option:

```bash
python examples/train_rbf_gp.py --help
```

- `--steps`: 最大optimizer step（既定10,000）
- `--min-steps`: early stopping前に最低限行うstep（既定5,000）
- `--length-scale`: RBFの滑らかさ
- `--signal-scale`: GP functionのscale（`--signal-std`もalias）
- `--observation-noise`: covariance対角に加えるnoise standard deviation
- `--grid-points`: GP sample grid数。Context最大数 + Query数より多くする
- `--seed`: model初期値、train sampling、固定bankを再現する基準seed
- `--no-plot`: PNGを生成しない

kernel parameterを変更したrunは別の関数分布を学びます。既定checkpointとの直接比較や、既定GUI modeへの読み込みではparameterを混ぜないでください。

## 生成物

`outputs/rbf_gp/`へ保存します。

| file | 内容 |
|---|---|
| `best_model.pt` | validationで選んだmodel、kernel、seed、test gate summary |
| `results.json` | config、環境、学習履歴、validation/testの全metric、gate |
| `learning_curve.png` | train/validation NLLとvalidation RMSE |
| `context_sweep.png` | Context 3/5/10/20で3方式のRMSE比較 |
| `predictions.png` | held-out GP function、Context、CNP、GP posterior |

checkpointの最小schemaは次の通りです。

```text
format_version = 1
benchmark = "rbf_gp"
model_config
model_state_dict
step
x_range
grid_points
kernel
training_config
gate_passed
test_gate
test_summary
```

## 収録run結果

repositoryに収録した既定設定runは、early stoppingせず10,000 stepまで実行され、最良validation checkpointも最終stepの10,000でした。

実行環境はPython 3.9.6、PyTorch 2.8.0、macOS 15.7.4 arm64、Apple MPSです。環境情報は`results.json`にも保存しています。

| 学習量 | 実測値 |
|---|---:|
| optimizer step（best / final） | 10,000 / 10,000 |
| function episode | 640,000 |
| Context observation | 7,361,822 |
| loss対象Query | 20,480,000 |
| wall time | 91.12秒 |
| best validation RMSE | 0.6604 |

held-out test 256関数における、Contextと重複しないQueryのfunction-macro RMSEは次の通りです。

| 方式 | Context 3 | Context 5 | Context 10 | Context 20 |
|---|---:|---:|---:|---:|
| CNP | 0.7712 | 0.6893 | 0.5941 | 0.5283 |
| Analytic GP posterior | 0.7109 | 0.5645 | 0.2823 | 0.0706 |
| Context mean | 0.9647 | 0.9127 | 0.8830 | 0.8640 |

事前固定gateは両方とも`PASS`でした。

- Context 3から20へのCNP RMSE改善: **31.5%**（必要値20%以上）
- Context 10でContext meanに対するCNP RMSE改善: **32.7%**（必要値25%以上）

これにより、modelがContextの追加情報を利用し、位置に依存しない平均値baselineより良く未知点を補間できることは確認できました。一方、Context 20ではCNP `0.5283`に対してkernelを知るGP posteriorは`0.0706`であり、近似精度には大きな差があります。また、validation RMSEは9,500 stepの`0.6633`、9,750 stepの`0.6631`から10,000 stepの`0.6604`まで改善を続け、best checkpointが学習上限に到達しました。したがって、このrunは**基本的な正常性確認には成功**しましたが、十分な収束や原論文の再現を示すものではありません。公式notebookの200,000 iterationとも学習量が大きく異なります。

収録fileのSHA-256は次の通りです。再学習でfileを上書きした場合は変化します。

| file | SHA-256 |
|---|---|
| `best_model.pt` | `b8a9c688b7d85ecf47e29e0ba79b2c5213733e70845b1c49391d6abf85049901` |
| `results.json` | `eaa09546fa8c9b52308e754715e92ba9a1691b54cc68b0531e04d501f4dd4067` |
| `learning_curve.png` | `9f906bd90880e49d5efd24cd9b3e96a1feda19705892f74a359703de628d89d0` |
| `context_sweep.png` | `d1e5c696f5f51189c9710a0fa936104ba82aa1a4026259aa95644f99c7573c1d` |
| `predictions.png` | `709bb141feca97a73cdb1166554577abb2cee480e0191c649afbfc5779357144` |

## GUIで試す

学習後にnative appを起動し、experiment modeで`RBF-GP`を選択します。

```bash
python apps/sanity_desktop.py \
  --checkpoint outputs/sanity/best_models.pt \
  --rbf-checkpoint outputs/rbf_gp/best_model.pt
```

GUIではheld-out GP function、Context数、Context抽選seed、任意のQuery Xを変更し、CNP predictionとanalytic GP posteriorを同じplotで比較できます。ブラウザserverは使いません。

## 読み方と限界

- RBF-GPは多様ですが、1つの固定kernelから生成したsynthetic familyです。実心拍の非定常性、運動、睡眠、薬、sensor artifact、個人差は表現していません。
- CNPの出力はQueryごとのfactorized Gaussianです。複数Query間の共分散を直接出力しないため、表示bandはpointwise intervalであり、関数全体の同時信頼帯ではありません。
- このbenchmarkはdomain内の欠損補間です。未来の時刻への外挿は別problemとして評価する必要があります。
- GP posteriorとの差は、network容量・最適化・有限学習・CNPのfactorizationによる近似差を含みます。
- Context数が増えても、1個のfunction drawではRMSEが必ず単調に下がるとは限りません。判定は256 functionのmacro平均で行います。
