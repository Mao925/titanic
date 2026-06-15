# Titanic 生存予測モデル構築プロジェクト

## プロジェクト概要

Kaggle の Titanic データセットを用いて、生存予測モデルを構築したプロジェクトです。

本プロジェクトでは、単にスコアを追うだけではなく、以下の観点を重視して検証を行いました。

- 欠損値処理
- 特徴量エンジニアリング
- 複数モデルの比較
- 交差検証（CV）と Kaggle Public LB の乖離分析
- 過学習を抑えるためのハイパーパラメータ調整

最終的には、RandomForestClassifier において、**複雑なモデルではなく、浅い木 + やや大きめの leaf による保守的な設定**が最も良い Public LB を記録しました。

---

## 使用データ

Kaggle Titanic: Machine Learning from Disaster

- `train.csv`
- `test.csv`

目的変数は `Survived` です。

---

## 前処理・特徴量作成

主に以下の前処理と特徴量作成を行いました。

### 欠損値処理

- `Embarked`
  - 最頻値で補完
- `Fare`
  - 中央値で補完
- `Age`
  - `Sex × Pclass` ごとの中央値で補完
- `Cabin`
  - 欠損有無を `HasCabin` として特徴量化
  - 先頭文字を `Deck` として抽出

### 作成した特徴量

- `Title`
  - `Name` から敬称を抽出
  - `Mr`, `Mrs`, `Miss`, `Master`, `Rare` などに整理
- `FamilySize`
  - `SibSp + Parch + 1`
- `IsAlone`
  - 一人乗船かどうか
- `IsChild`
  - 16歳未満かどうか
- `HasCabin`
  - Cabin 情報の有無
- `Deck`
  - Cabin の先頭文字
- `TicketGroupSize`
  - 同一チケットの人数
- `FarePerPerson`
  - `Fare / TicketGroupSize`

### カテゴリ変数処理

以下のカテゴリ変数を one-hot encoding しました。

- `Sex`
- `Embarked`
- `Title`
- `Deck`

---

## モデル検証の流れ

### 1. Logistic Regression / Decision Tree

まずはシンプルなモデルでベースラインを確認しました。

| Model | CV Accuracy |
|---|---:|
| LogisticRegression | 約 0.79 |
| DecisionTree | 約 0.778 |

線形モデルでも一定の精度は出たものの、特徴量の非線形な関係を扱うため、木系モデルを中心に検証する方針としました。

---

### 2. RandomForest 初期モデル

RandomForest を導入したところ、CV が改善しました。

| Model | CV Accuracy |
|---|---:|
| RandomForest 初期 | 約 0.816 |

この時点で、Titanic においては RandomForest が有力であると判断しました。

---

### 3. RandomForest conservative baseline

過学習を避けるため、木の深さを抑えた conservative な RandomForest を作成しました。

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=4,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

| Model | CV Accuracy | Public LB |
|---|---:|---:|
| RF conservative baseline | 0.82827 | 0.78947 |

このモデルが当初のベストモデルとなりました。

---

## ハイパーパラメータ調整の検証

### 当初の仮説

当初は、README に示した高性能な RandomForest をベースラインとして、`max_depth` と `min_samples_leaf` を調整することで、過学習を防ぎつつ精度改善を目指しました。

検証対象は主に以下です。

- `max_depth`
- `min_samples_leaf`

---

## 検証1: CV 上の best_tuned モデル

GridSearch の結果、CV 上では以下のようなモデルが高スコアとなりました。

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

| Model | CV Accuracy | Public LB |
|---|---:|---:|
| RF conservative baseline | 0.82827 | 0.78947 |
| RF best_tuned | 約 0.84286 | 0.78468 |

CV は改善したものの、Public LB は悪化しました。

### 考察

この設定は、過学習防止ではなく、むしろモデルを複雑化する方向でした。

- `max_depth: 4 → 8`
  - 木が深くなり、細かい分岐を拾いやすくなる
- `min_samples_leaf: 4 → 3`
  - 少数サンプルの葉を許容し、ノイズを拾いやすくなる

そのため、CV では高く見えても、Public LB では汎化性能が落ちたと考えられます。

### 学び

Titanic のような小規模データでは、CV スコアの最大化がそのまま Public LB の改善につながるとは限りません。

---

## 検証2: max_depth=4 固定で min_samples_leaf を調整

次に、`max_depth=4` を固定し、`min_samples_leaf` を大きくする方向を試しました。

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=8,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

| Model | Public LB |
|---|---:|
| max_depth=4, min_samples_leaf=4 | 0.78947 |
| max_depth=4, min_samples_leaf=8 | 0.78947 |

スコアは横ばいでした。

### 考察

`min_samples_leaf=8` は、`4` よりも保守的な設定であり、過学習抑制の方向性としては正しいです。

ただし、Public LB は改善しなかったため、`max_depth=4` のままでは限界がある可能性が見えました。

---

## 検証3: max_depth=3, min_samples_leaf=8

次に、木の深さをさらに浅くして、`max_depth=3` を試しました。

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=8,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

| Model | Public LB |
|---|---:|
| max_depth=4, min_samples_leaf=8 | 0.78947 |
| max_depth=3, min_samples_leaf=8 | 0.79186 |

Public LB が `0.79186` となり、ベストスコアを更新しました。

### 考察

`max_depth=3` に下げることで、木の分岐がより単純になり、Titanic の test data に対する汎化性能が改善したと考えられます。

この結果から、今回のデータでは複雑な木よりも、浅い木を多数組み合わせる方が有効である可能性が高いと判断しました。

---

## 検証4: max_depth=3 固定で min_samples_leaf を微調整

`max_depth=3` が有望だったため、次に `min_samples_leaf` の近傍を確認しました。

| max_depth | min_samples_leaf | Public LB |
|---:|---:|---:|
| 3 | 6 | 0.78947 |
| 3 | 7 | 0.79186 |
| 3 | 8 | 0.79186 |
| 3 | 9 | 0.78947 |

`min_samples_leaf=7` と `8` が同率でベストとなりました。

### 考察

`min_samples_leaf=6` ではやや複雑で、`9` ではやや単純化しすぎた可能性があります。

今回の結果から、`max_depth=3` の場合、`min_samples_leaf=7〜8` が最もバランスの良い範囲だと判断しました。

---

## 検証5: max_depth=2

さらに単純化した場合を確認するため、`max_depth=2` も試しました。

| Model | Public LB |
|---|---:|
| max_depth=3, min_samples_leaf=8 | 0.79186 |
| max_depth=2, min_samples_leaf=8 | 0.78229 |

`max_depth=2` では大きく悪化しました。

### 考察

`max_depth=2` ではモデルが単純すぎて、必要な分岐を表現できていない可能性が高いです。

したがって、今回のデータでは以下のように判断できます。

- `max_depth=2`
  - 過小適合気味
- `max_depth=3`
  - 最も汎化性能が高い
- `max_depth=4`
  - 悪くないが、やや複雑
- `max_depth=8`
  - CV では高いが、Public LB では過学習気味

---

## 最終ベースライン

今回の検証を踏まえ、最終的なベースラインは以下とします。

### Baseline A

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=7,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

### Baseline B

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=8,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

| Model | Public LB |
|---|---:|
| RF baseline old: depth=4, leaf=4 | 0.78947 |
| RF baseline new: depth=3, leaf=7 | 0.79186 |
| RF baseline new: depth=3, leaf=8 | 0.79186 |

最終的には、`max_depth=3, min_samples_leaf=7 or 8` を新しいベースラインとします。

---

## 提出履歴

| No. | 内容 | Public LB |
|---:|---|---:|
| 1 | 初期モデル | 0.77272 |
| 2 | 特徴量追加・モデル改善 | 0.77511 |
| 3 | 改善版 RandomForest | 0.77990 |
| 4 | conservative RF baseline | 0.78947 |
| 5 | best_tuned: depth=8, leaf=3 | 0.78468 |
| 6 | depth=4, leaf=8 | 0.78947 |
| 7 | depth=3, leaf=8 | 0.79186 |
| 8 | depth=3, leaf=7 | 0.79186 |
| 9 | depth=2, leaf=8 | 0.78229 |
| 10 | depth=3, leaf=6 | 0.78947 |
| 11 | depth=3, leaf=9 | 0.78947 |

---

## 今回の重要な学び

### 1. CV 最良モデルが Public LB 最良とは限らない

`max_depth=8, min_samples_leaf=3` は CV 上では良かったものの、Public LB では悪化しました。

これは、CV に対する過適合、または Titanic の小規模データ特有の分割依存が影響している可能性があります。

---

### 2. 過学習防止の方向性を誤るとスコアは悪化する

当初の tuned モデルは、過学習防止を目的としていたにもかかわらず、実際には以下のようにモデルを複雑化していました。

- `max_depth` を大きくする
- `min_samples_leaf` を小さくする

この方向は、RandomForest において過学習を促進する可能性があります。

---

### 3. 今回は「浅い木 + やや大きめの leaf」が有効だった

最も良かった設定は以下でした。

```text
max_depth = 3
min_samples_leaf = 7 or 8
```

木を深くするよりも、浅い木で安定した分岐を行う方が Public LB では良い結果となりました。

---

### 4. 過小適合にも注意が必要

`max_depth=2` では Public LB が `0.78229` まで低下しました。

そのため、単純化すればするほど良いわけではなく、今回のデータでは `max_depth=3` がちょうど良い複雑さだったと考えられます。

---

## 現時点の結論

現時点では、以下のモデルを最終ベースラインとします。

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=8,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

または、

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=7,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
```

Public LB はどちらも `0.79186` であり、旧ベースライン `0.78947` を上回りました。

---

## 今後の改善方針

今後スコア改善を狙う場合は、単純に RandomForest を複雑化するのではなく、以下の方向を優先します。

### 1. 特徴量の見直し

- `Title` の分類方法の再検討
- `TicketGroupSize` の扱い
- `FarePerPerson` の扱い
- `Deck` / `HasCabin` の有効性検証
- 家族単位の生存傾向の分析

### 2. 検証設計の改善

- Repeated Stratified KFold による安定性確認
- 複数 random_state での CV 平均確認
- CV と Public LB の乖離が大きい設定を避ける

### 3. モデルの比較

- Logistic Regression
- RandomForest
- GradientBoosting
- XGBoost
- Voting / Blending

ただし、Titanic はデータ数が少ないため、複雑なモデルほど良いとは限りません。

---

## まとめ

今回の検証では、当初の conservative RandomForest baseline をさらに単純化することで、Public LB を更新できました。

最も重要な学びは、CV スコアを盲目的に最大化するのではなく、モデルの複雑さと汎化性能のバランスを見る必要があるという点です。

最終的には、以下の方針が有効でした。

```text
木を深くしすぎない。
葉を小さくしすぎない。
CV最高値ではなく、汎化しそうな保守的設定を選ぶ。
```

この方針により、Public LB は以下のように改善しました。

```text
旧ベースライン: 0.78947
新ベースライン: 0.79186
```
