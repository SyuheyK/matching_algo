# 自社DSP ロードマップ：広告絞り込み〜入札最適化

決済後画面に媒体広告を出し、決済ユーザーの文脈に応じて「どの広告を出すか（絞り込み）」と
「いくら入札するか（入札）」を自前で最適化する自社DSPの構築ロードマップ。
Bypass（外部DSP）は全面リプレイス対象。

システムは2層に分離する。本ドキュメントは主に**絞り込み層（Contextual Bandit）**を軸に据えつつ、
接続先である**入札層**の位置づけまで含めて方向性を示す。

- 絞り込み層：どの広告を入札候補に残すか（探索・コールドスタート対応）
- 入札層：残した候補にいくら入札するか（予算制約・オークション）

---

## ① アルゴリズムの候補選定

### 1.1 絞り込み層の候補

| 候補 | 位置づけ | 採否の目安 |
|---|---|---|
| **LinUCB（disjoint / hybrid）** | 線形報酬を仮定し、信頼上界で探索。実装・運用・解釈のバランス型 | 第一候補。まずここから |
| **Linear Thompson Sampling（LinTS）** | ベイズ的に事後分布からサンプリングして探索 | 対抗候補。実運用で安定・高性能な報告が多い |
| **Logistic Bandit / GLM Bandit** | 報酬が二値（クリック/CV）で線形が歪む場合の一般化線形モデル | クリック率の非線形性が問題化したら移行 |
| **Neural Bandit** | 文脈と報酬の関係が非線形で複雑な場合 | データ量が十分に増えてから検討。初手では過剰 |

選定方針：報酬（クリック/CV）が文脈特徴の線形和で近似でき、交互作用は特徴設計で明示的に
与えられる規模のため、**LinUCB を主軸、LinTS を比較対象**とする。両者は
共通の線形リッジ回帰基盤（$A=D^\top D+\lambda I$）を共有でき、探索の与え方（UCB か事後サンプリングか）
だけが異なるため、片方の実装からもう片方へ低コストで拡張できる。

### 1.2 disjoint と hybrid の選択

- **disjoint**：腕（広告）ごとに独立した係数。腕をIDで持つと新規広告に履歴がなく学習できない。
- **hybrid**：全腕共通の係数 + 腕固有の係数。共通部分の学習が新規腕にも転用され、
  コールドスタートに効く。広告が頻繁に追加される本件と適合する。

方針：腕を**特徴ベクトルで表現する共有特徴空間**を採り、新規広告も特徴経由で価値推定できる形にする。
発展として hybrid（共通項＋広告固有項）を導入する。

### 1.3 入札層の候補（別系統）

絞り込み層とは独立に、以下を組み合わせる。

- **最適入札式（線形入札 / 非線形入札）**：期待価値から入札額を導く。予算制約下では
  価値そのままではなく制約を織り込んだ入札にする（双対変数による調整）。
- **予算ペーシング（budget pacing）**：日次予算を時間配分し、序盤で使い切る/余らせるを防ぐ。
  フィードバック制御（PID等）や smart pacing 系。
- **bid shading**：first-price オークションで、勝てる範囲で札を抑える。勝率・落札価格分布の推定を伴う。

注意：RTB は純粋な contextual bandit 単独では閉じない。予算の時間配分と競合の動的入札を
bandit だけでは扱えないため、絞り込み（bandit）と入札（最適化＋制御）を分離して設計する。

---

## ② ロジック作成

### 2.1 全体フロー

```
bid request（決済文脈 + 媒体シグナル）
      │
 ① 絞り込み層  ← Contextual Bandit（LinUCB/LinTS, 共有特徴空間）
    広告群 → 文脈に応じた上位K件へ、探索込みで選択
      │
 ② 入札層      ← 最適入札式 + 予算ペーシング + bid shading
    候補の期待価値 → bid price
      │
 bid response →（落札すれば）配信 → imp/click/CV
      │
 報酬フィードバック → ① へ（落札バイアス補正を考慮）
```

### 2.2 絞り込み層のロジック設計

**文脈ベクトル $x(\text{context}, \text{ad})$ の構成**
決済文脈と広告特徴、および両者の交差を1本の特徴ベクトルに畳み込む。線形モデルは
交互作用を自動では捉えないため、効きどころの交差を明示的に設計する。

- 決済文脈：車種、地域、地域タイプ（都市/観光）、価格帯、レンタル日数
- 広告特徴：広告カテゴリ、新規フラグ
- 交差：地域一致、観光地×スポット/レストラン、車種×グッズ適合、車種×カテゴリ、地域×カテゴリ

**更新式（リッジ回帰のオンライン更新）**
$A \leftarrow A + xx^\top,\quad b \leftarrow b + r\,x,\quad \theta = A^{-1}b$。
$A^{-1}$ は Sherman-Morrison で逐次更新し、毎ステップの逆行列計算を避ける（レイテンシ制約対応）。

- LinUCB のスコア：$\theta^\top x + \alpha\sqrt{x^\top A^{-1} x}$
- LinTS のスコア：$\tilde\theta^\top x,\ \ \tilde\theta \sim \mathcal{N}(\theta, v^2 A^{-1})$

**探索強度**
$\alpha$（LinUCB）または $v$（LinTS）が探索と活用のトレードオフを制御。
交差項を増やすほど次元が増え、regret が縮むまでのサンプル数が増える。最小構成から始め、
ログを見て交差を追加する反復設計とする。

### 2.3 報酬設計（要注意点）

RTB では入札に勝って初めて imp/click/CV が観測でき、負けたインプレッションの結果は不明。
つまり報酬は「選択→入札→落札→結果」の後段でしか得られず、**落札に依存したセレクションバイアス**が乗る。

- 対処案A：落札ログのみで学習し、傾向スコア重み付け（IPS 等）でバイアス補正
- 対処案B：報酬を落札から切り離し、出せた場合の CTR/CVR を学習対象にして選択層を入札勝敗から独立させる
- 遅延報酬（クリックやCVが遅れて届く）の扱いも設計に含める

### 2.4 入札層のロジック設計

- 期待価値 $=$ 予測CTR $\times$ 予測CVR $\times$ コンバージョン価値
- 入札額 $=$ 期待価値 $\times$ ペーシング係数 $\times$ shading 係数
- 予算制約下では、双対変数（ラグランジュ乗数）で「価値あたりコスト」を調整し、
  予算内でKPIを最大化する定式化に落とす

---

## ③ 実装

### 3.1 段階的な実装計画

1. **合成データでのプロトタイプ（実施済み）**
   - 車種3 × 千葉5都市の商品、広告約50件、1,000セッションのアクセスログ
   - LinUCB（共有特徴空間, 45次元, Sherman-Morrison更新）＋報酬シミュレータ
   - semi-bandit のオンライン評価で reward@K が oracle 近傍へ収束することを確認済み
2. **絞り込み層の拡張**
   - LinTS 実装を追加し、LinUCB と regret・CTR で比較
   - hybrid 化（共通項＋広告固有項）でコールドスタートを検証
   - 探索強度・特徴次元の感度分析、非定常対応（時間減衰・忘却係数）
3. **報酬パイプラインの現実化**
   - 報酬シミュレータを実クリック/CVログに差し替え
   - 落札バイアス補正（IPS 等）とオフライン評価（過去ログでの反実仮想評価）を導入
4. **入札層の実装**
   - 最適入札式 + 予算ペーシング + bid shading を実装
   - 絞り込み層の出力（期待価値）を入札層の入力に接続
5. **システム統合**
   - bid request/response のI/O、レイテンシ設計（数十ミリ秒応答）
   - オンライン学習の更新頻度、モデル配信、監視・ロールバック

### 3.2 実装上の技術的論点

- **レイテンシ**：秒間多数の bid request に数十ミリ秒で応答。特徴生成とスコアリングの軽量化、
  $A^{-1}$ の逐次更新、候補の事前フィルタ（地理・カテゴリのハードフィルタ）で腕数を削減
- **オフライン評価**：オンライン投入前に、過去ログで反実仮想評価（replay 系の手法）を行う
- **監視**：regret・CTR・予算消化・勝率のダッシュボード化、探索比率の可観測性

### 3.3 現状の実装成果物

- `data_spec.md`：データ仕様
- `products.csv` / `ads.csv` / `access_log.csv`：合成データ
- `linucb.py`：LinUCB 絞り込みロジック（報酬シミュレータ・評価ループ込み）
- `bandit_regret.csv`：学習推移（regret 曲線）

---

## ④ 参考文献

### 絞り込み層（Contextual Bandit）

- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation.* Proceedings of the 19th International Conference on World Wide Web (WWW '10), 661–670. — LinUCB（disjoint / hybrid）の原典。
- Chu, W., Li, L., Reyzin, L., & Schapire, R. E. (2011). *Contextual Bandits with Linear Payoff Functions.* Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS 2011), 208–214. — 線形報酬 bandit の理論的裏付け。
- Agrawal, S., & Goyal, N. (2013). *Thompson Sampling for Contextual Bandits with Linear Payoffs.* Proceedings of the 30th International Conference on Machine Learning (ICML 2013), 127–135. — LinTS の理論と regret 限界。
- Li, L., Chu, W., Langford, J., & Wang, X. (2011). *Unbiased Offline Evaluation of Contextual-Bandit-based News Article Recommendation Algorithms.* Proceedings of the 4th ACM International Conference on Web Search and Data Mining (WSDM '11), 297–306. — bandit のオフライン評価。

### 入札層（RTB・入札最適化・予算ペーシング）

- Wang, J., Zhang, W., & Yuan, S. (2017). *Display Advertising with Real-Time Bidding (RTB) and Behavioural Targeting.* Foundations and Trends in Information Retrieval, 11(4–5), 297–435. — RTB全体（価値予測・入札戦略・予算ペーシング・オークション）を体系化したサーベイ。自社DSP設計の起点。
- Zhang, W., Yuan, S., & Wang, J. (2014). *Optimal Real-Time Bidding for Display Advertising.* Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '14), 1077–1086. — 予算制約下の最適入札式の基礎。
- Cai, H., Ren, K., Zhang, W., Malialis, K., Wang, J., Yu, Y., & Guo, D. (2017). *Real-Time Bidding by Reinforcement Learning in Display Advertising.* Proceedings of the 10th ACM International Conference on Web Search and Data Mining (WSDM '17), 661–670. — 入札を逐次意思決定（MDP）として定式化。
- Jin, J., Song, C., Li, H., Gai, K., Wang, J., & Zhang, W. (2018). *Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Advertising.* Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM '18). — 競合をマルチエージェント強化学習で扱う。
- Xu, J., Lee, K.-c., Li, W., Qi, H., & Lu, Q. (2015). *Smart Pacing for Effective Online Ad Campaign Optimization.* Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15). — 予算ペーシング（smart pacing）。
- Zhang, W., Zhou, T., Wang, J., & Xu, J. (2016). *Bid-aware Gradient Descent for Unbiased Learning with Censored Data in Display Advertising.* KDD. — 落札打ち切り（censored）データの不偏学習、セレクションバイアス対処。

### データセット・ベンチマーク

- Zhang, W., Yuan, S., Wang, J., & Shen, X. (2014). *Real-Time Bidding Benchmarking with iPinYou Dataset.* arXiv:1407.7073. — RTB 公開データセット。

### 補助リソース

- wnzhang/rtb-papers（GitHub）: RTB 関連論文・サーベイの網羅的コレクション。
