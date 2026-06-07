# Titanic 生存予測モデル構築プロジェクト

## プロジェクト概要

KaggleのTitanicデータセットを用いた生存予測モデルの構築プロセスと、特徴量・モデル検証の履歴をまとめたリポジトリです。最終的にXGBoostを用いて交差検証（Cross Validation）で約84.4%の精度（Accuracy）を達成し、KaggleのPublic Scoreでは保守的なRandom Forestを用いて0.78947を記録しました 。

## モデル選定の背景

Titanicの生存予測においては、Logistic Regressionや単体のDecisionTreeよりも、RandomForestやGradientBoostingなどのアンサンブル学習モデルが適しています 。その理由は主に以下の3点です。

* **複雑な非線形関係の把握**: 「男性かつ高い運賃」や「女性かつ1等客室」など、特徴量同士の複雑な組み合わせや非線形な関係性を捉えやすい特徴があります 。


* **過学習の抑制**: 複数の決定木を組み合わせることで、単体のDecisionTreeが陥りやすい訓練データのノイズ学習を防ぐことができます 。


* **データ構造との相性**: Pclass、Sex、Embarkedといったカテゴリやフラグ系の「ルールベースに近い構造」を持つデータから、分岐点や有効な組み合わせを自動で発見しやすいためです 。



## 特徴量エンジニアリング（Feature Engineering）

### 成功した特徴量・前処理

* **FamilySize**: SibSpとParchに1を足して算出し、家族連れか一人旅かを表現しました 。


* **IsAlone**: FamilySizeが1の場合を1とし、完全な一人客である状態を表現しました 。


* **IsChild**: 16歳未満（Age < 16）を1とし、子ども優先避難の影響をモデルに組み込みました 。


* **Ageの欠損値補完**: 性別と客室等級の組み合わせ（Sex × Pclass）ごとの中央値で補完し、より自然な年齢分布を維持しました 。


* **HasCabin**: 欠損の多いCabin列について、記録の有無そのものが上位客室利用者を示す可能性があるため、0/1のフラグとして追加しました 。


* **TicketGroupSize**: 同じTicket番号を持つ人数を算出し、FamilySizeでは捉えきれない団体・家族行動の情報を追加しました 。


* **Deck**: Cabinの先頭文字（A〜G）を抽出し、欠損値はUnknownとして扱いました 。これにより客室の位置や等級情報を反映させることができ、精度向上に寄与しました 。


* **前処理の統一化**: 汎化性能向上のため、trainデータとtestデータを結合し、欠損値補完やダミー化を同一手順で実行しました 。



### 失敗・効果が薄かった施策

* **Title特徴量（Nameからの敬称抽出）**: SexやAgeの情報と重複してしまい、新しい情報を追加できずCV精度が低下（0.8316→0.8283）しました 。


* **Age/FareのビニングやPclass×Sexの作成**: RandomForestが自動で学習できる数値分岐を人為的に区間化したことで情報量が減少した可能性があります 。結果として精度が低下（0.8272）しました 。


* **RandomForestの過学習抑制**: max_depthを浅くしmin_samples_leafを増やす調整を行いました 。しかし、モデルが十分なパターンを学習できなくなり精度が低下（0.8249）しました 。



## モデルの変遷と評価スコア

各試行における5-fold Cross Validation（CV）の最高精度の推移です。

| モデル | 施策内容 | CV Accuracy |
|---|---|---:|
| Random Forest | 初期（特徴量追加：FamilySize, IsAlone, IsChild） | 0.8320 |
| Gradient Boosting | モデル変更・ハイパーパラメータ調整（lr=0.1, max_depth=4等） | 0.8328 |
| Gradient Boosting | Age補完改善＋HasCabin＋TicketGroupSize追加 | 0.8362 |
| Gradient Boosting | さらにDeck特徴量を追加 | 0.8373 |
| XGBoost | Deck・HasCabin・TicketGroupSizeを活用しGridSearch実行 | 0.8440 |

* **Kaggle Public Score 最高値**: 0.78947 


* **最高スコアモデルの構成**: Sex × PclassによるAge補完、Title、FamilySize、IsAlone、IsChild、HasCabin、Deck、TicketGroupSize、FarePerPersonを利用した保守的なRandom Forest 。



## 考察と今後の課題

* **モデルの特性**: GradientBoostingやXGBoostは誤分類データを重点的に修正できるため、Titanicのような小規模データセットでの細かなパターン学習や、複数特徴量の相互作用の利用に優れていました 。


* **ドメイン知識の重要性**: 「客室等級が高いほど救命ボートへアクセスしやすい（Deck / HasCabin）」や「子どもや家族連れの避難傾向（IsChild / FamilySize）」など、仮説に基づく特徴量作成が精度向上に直結しました 。


* **汎化性能の課題**: CVスコアとKaggleのPublic Score間に乖離が見られます 。今後は大幅な特徴量変更を行うよりも、最高スコアを出した保守的なRandom Forestモデルをベースに、過学習を抑えるためのパラメータ（max_depthやmin_samples_leafなど）を微調整していく方針が妥当と考えられます 。
