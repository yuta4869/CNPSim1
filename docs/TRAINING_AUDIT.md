# 現在の学習量、結果、既知課題

## 結論

現在のcheckpointは、CNPによる心拍欠損補間のデータ処理、学習、未知患者評価、GUI推論が一通り動くことを確認するPoCとしては十分です。一方、「CNPが心拍補間に適している」「未知患者へ安定して一般化する」と結論するには不十分です。

実行全体はstep 1,000でearly stopしましたが、アプリと評価に採用したのはvalidation MAEが最小だったstep 500です。step 500以降は改善していないため、現在の問題を単純なstep不足だけで説明することはできません。

## データと患者分割

- BIDMC v1.0.0: 53 records / 46 unique patient IDs
- 1 recordは481点、HRは1 Hz
- 各recordの先頭480点を使用
- 患者非重複split、seed 7
  - train: 32患者 / 36 records / 144 windows
  - validation: 7患者 / 10 records / 40 windows
  - test: 7患者 / 7 records / 27 windows
- 120秒の非重複windowを1つのCNP functionとして扱う
- trainの固有HR値: 17,280点 = 17,280秒 = 4.8時間
- 時刻: 各windowを`[-1, 1]`へ正規化
- HR: train患者だけで標準化
  - mean: 90.777025 bpm
  - population std: 14.289872 bpm

`bidmc_27`の207–210秒に4点の自然欠損があります。正解がないため、その欠損を含む120秒test windowを1本除外しました。学習・評価targetは、正解のあるHRを人工的に隠して作っています。

患者IDはsplitと患者一様samplingにだけ使用し、CNP入力には入れていません。年齢、性別、PPG、呼吸信号なども現在のモデル入力には含めていません。

## 1回の学習処理

各optimizer stepで欠損条件を1つ選び、同じ条件でbatch size 16を構築します。各batch itemでは、train患者を一様に選び、その患者のwindowを1本選びます。

| 条件 | 選択確率 | Context | Target |
|---|---:|---:|---:|
| Random 10% | 約23.3% | 108 | 12 |
| Random 30% | 約23.3% | 84 | 36 |
| Random 50% | 約23.3% | 60 | 60 |
| Gap 5秒 | 10% | 115 | 5 |
| Gap 15秒 | 10% | 105 | 15 |
| Gap 30秒 | 10% | 90 | 30 |

Randomではwindow両端をContextに残します。Gapでは左右に最低5点ずつContextを残します。Target HRはCNP入力から完全に除外し、lossの正解としてだけ使用します。

```text
Context (time, HR) --Encoder--> 64次元表現 --mean--> r
                                                   |
Query time ----------------------------------------+--> Decoder --> mean, scale
Target HR --------------------------------------------------------> Gaussian NLL
```

モデルはhidden 64、representation 64のdeterministic CNPで、trainable parameterは17,026個です。全Context表現を単純平均します。Attention、latent variable、CNN、RNNはありません。

最適化設定:

- Adam、learning rate `1e-3`
- Gaussian negative log-likelihood
- gradient global norm clipping `1.0`
- batch size 16
- validation every 100 steps
- 最大1,500 steps、minimum 800、patience 5
- 6欠損条件の患者macro MAE平均でcheckpoint選択

収録runの環境:

- Python 3.10.20
- PyTorch 2.13.0
- macOS 15.7.4 arm64
- Apple MPS

依存versionはlockしておらず、PyTorchやMPSの実行環境差により、同じseedでもbest stepや評価値の完全一致は保証されません。

## 採用checkpointが実際に受けた学習量

seed済みsampling系列をstep 500まで再構成した結果です。

| 条件 | Updates | Window tasks | Context usages | Target loss points |
|---|---:|---:|---:|---:|
| Random 10% | 127 | 2,032 | 219,456 | 24,384 |
| Random 30% | 108 | 1,728 | 145,152 | 62,208 |
| Random 50% | 112 | 1,792 | 107,520 | 107,520 |
| Gap 5秒 | 51 | 816 | 93,840 | 4,080 |
| Gap 15秒 | 63 | 1,008 | 105,840 | 15,120 |
| Gap 30秒 | 39 | 624 | 56,160 | 18,720 |
| 合計 | **500** | **8,000** | **727,968** | **232,032** |

- 144 train windowsはすべて少なくとも一度使用
- window当たり平均55.6回使用。患者一様samplingのため実際は8〜81回
- Target候補16,992位置のうち16,944位置、約99.72%を少なくとも一度Targetとして使用
- 全17,280 HR点を少なくとも一度Contextとして使用

これは232,032個の独立ラベルがあるという意味ではありません。同じ17,280個のHR値へ異なる人工maskをかけた再利用回数です。

run自体はstep 1,000まで進み、16,000 window tasks、475,104 Target usagesを処理しました。しかしstep 600–1,000はvalidation MAEを更新できず、最後にstep 500をreloadしたため、アプリの重みには含まれません。

## 学習履歴

| Step | Validation MAE (bpm) | Validation NLL |
|---:|---:|---:|
| 1 | 10.278 | 1.270 |
| 100 | 1.397 | -0.480 |
| 200 | 1.286 | -0.463 |
| 300 | 1.226 | -0.362 |
| 400 | 1.325 | -0.552 |
| **500** | **1.175** | -0.433 |
| 600 | 1.309 | -0.390 |
| 700 | 1.198 | -0.363 |
| 800 | 1.208 | -0.404 |
| 900 | 1.277 | -0.473 |
| 1,000 | 1.266 | -0.529 |

best後に5回連続でMAEが改善せずearly stopしました。記録されたgradient normは11点中10点でclip閾値1.0を超えているため、学習率と最適化安定性の比較は今後必要です。

## Test結果

testは未知患者7人、27 windows、6条件を各5回評価しています。患者macro MAEは次のとおりです。

| 人工欠損 | CNP | 線形補間 | 最近傍 | Context平均 |
|---|---:|---:|---:|---:|
| Random 10% | 0.683 | 0.168 | 0.169 | 0.576 |
| Random 30% | 0.678 | 0.175 | 0.172 | 0.573 |
| Random 50% | 0.690 | 0.189 | 0.183 | 0.583 |
| Gap 5秒 | 0.694 | 0.222 | 0.204 | 0.613 |
| Gap 15秒 | 0.775 | 0.372 | 0.365 | 0.678 |
| Gap 30秒 | 0.683 | 0.528 | 0.524 | 0.589 |

6条件単純平均:

- CNP MAE: 0.700 bpm
- linear MAE: 0.276 bpm
- Context mean MAE: 0.602 bpm

CNPは全6条件で線形補間・最近傍・Context平均より悪い結果です。BIDMCの1 Hz HRは短時間では滑らかで、左右の観測点を直接結ぶ線形補間に有利です。基本CNPは全Contextを1つの平均表現に圧縮し、Query近傍を直接参照できないため、局所補間との構造的な不一致があります。

CNPの患者macro RMSEは0.873–0.981 bpmです。nominal 68% intervalのcoverageは96.4–98.5%、nominal 95%は99.8–100%、95% interval幅は約13.1–13.2 bpmでした。実誤差に比べて分布が広く、不確実性は未校正です。

## CNMP論文との比較

[RSS 2019論文](https://www.roboticsproceedings.org/rss15/p71.pdf)では、各iterationでデモを1本選び、同じデモからContextとQuery 1点をsampleしてGaussian NLLで1回更新します。論文本文には総iteration数は記載されていません。

| 論文実験 | 学習デモ数 | 記載された設定 | 総iteration |
|---|---:|---|---|
| 2D障害物回避 | 6軌跡×300点 | Context 1〜5、Encoder/Query各2層×128、Adam `1e-4` | 記載なし |
| 画像付き障害物回避 | 1,000軌跡 | CNN channels 8/16/32/64 | 記載なし |
| 実ロボット・parameterization | 8軌跡 | 物体高2×障害物高4 | 記載なし |
| 実ロボット・摂動適応 | 20軌跡 | 2容器×2カテゴリ×5重量 | 記載なし |

公式の[Keras汎用notebook](https://github.com/myunusseker/CNMP/blob/Keras/Conditional%20Neural%20Movement%20Primitives.ipynb)は6 train＋4 validation軌跡、各200点で100万stepを完走した出力を持ちます。論文の2D障害物回避[補足notebook](https://github.com/myunusseker/CNMP/blob/Keras/Example%20Experiments/Learning%202D%20Obstacle%20Avoidance/Learning%20of%202D%20obstacle%20avoidance%20skill%20shown%20in%20categorically%20different%20ways.ipynb)は200万iterationを設定していますが、保存出力はstep 10,000付近で中断しており、点数も論文の300ではなく200です。後発の[PyTorch再実装](https://github.com/myunusseker/CNMP/blob/95045db72e9f0233e7029993c1c668ff00dc2eb8/CNMP_PyTorch.ipynb)にも100万stepのループがありますが、保存notebookには完走出力がありません。

したがって、「論文では100万または200万iteration学習した」と断定することはできません。これらは公開コードの設定値です。

また、step数の単純比較もできません。公開コードは基本的に1 updateで1デモ・1 Queryですが、本実装は1 updateで16 windows・平均約464 Target点を使います。現在のbestは500 updatesですが232,032 Target loss pointsを処理しており、100万の単一Query updateとのTarget投入数の差は約4.3倍です。ただし多数Targetを平均して1回更新するため、optimizer update数は大幅に少ないままです。

## 既知課題

優先度順です。

1. **局所補間性能**: 全6条件で線形・最近傍・Context平均に敗けている。
2. **モデル構造**: mean aggregationのためQuery近傍のContextを直接参照できない。
3. **評価の不確実性**: 1固定split・1 seed、test患者7人だけで信頼区間がない。
4. **不確実性校正**: 予測区間が広すぎる。MAE用とNLL用checkpointも分離が必要。
5. **データ多様性**: trainは32患者、4.8時間で、短いICU記録だけ。
6. **欠損機構**: 自然欠損ではなく、ほぼすべて人工欠損。
7. **学習範囲**: 連続欠損は5・15・30秒のみ。GUIの31〜60秒はout-of-distribution。
8. **未来予測**: 現在は左右Contextのある補間専用で、未来予測を学習していない。
9. **入力特徴**: 患者属性、患者ID、PPG、活動状態などを使っていない。
10. **探索不足**: learning rate、model容量、window長、mask分布の系統的比較がない。
11. **再現環境**: dependency lockと決定論設定がなく、環境を跨ぐ数値の完全一致を保証しない。
12. **配布**: macOS `.app`はソースとPython環境を必要とする軽量launcherで、standalone bundleではない。
13. **ライセンス**: プロジェクトコードのライセンスが未選択。BIDMCのライセンスとは別に決定が必要。

## 次の実験

1. 患者GroupKFold 5-fold、各fold内の患者validation、3 seedsで再評価する。
2. Attentive CNP、local-context CNP、linear-residual CNPを基本CNPと比較する。
3. 固定mask bankを使い、checkpoint間で同一の欠損を評価する。
4. 最大5,000 steps程度を上限にearly stoppingし、learning rate `1e-4`, `3e-4`, `1e-3`を比較する。
5. GUI用途に合わせ、1〜60秒の連続分布からGapを学習する。
6. MAE、NLL、CRPS、coverage、interval width、calibration curveを報告する。
7. 線形、nearestに加え、spline、Kalman filter、Gaussian Process等をbaselineへ追加する。
8. より多数・長時間の患者データと外部datasetで検証する。

採用判断では、患者paired MAEで線形補間を改善し、その差の95% confidence intervalが0を跨がないことを最低条件とするのが妥当です。

## 成果物

- `outputs/bidmc/best_model.pt`
- `outputs/bidmc/results.json`
- `outputs/bidmc/example_gap30.png`

公開成果物のSHA-256:

```text
bd7dae05181d5bba8920dab65a81ef42a2dfb946288f5e315bb8d8ad30b454c7  best_model.pt
1952c080a21463d4f9513fa667716f06a2d163c5efef8dec692a2bb9ee187d57  results.json
e927043a84b4f460741e4af6056bdd840342d18efb3db1f32a5ccaf5453c6f2e  example_gap30.png
```

機械可読な全数値は`results.json`を正とします。
