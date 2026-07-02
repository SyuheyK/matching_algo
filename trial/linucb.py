"""
Linear Contextual Bandit (LinUCB, disjoint) による広告絞り込みロジック

役割:
  決済(purchase)イベントの文脈を受け取り、広告群(ads)から
  期待報酬の高い広告を上位K件に絞り込む。探索(UCB)込み。

構成:
  1. 文脈ベクトル x の構築          : 決済文脈 × 広告特徴 の交差を含む
  2. LinUCBPolicy                  : 腕(広告)ごとの線形モデル + UCB
  3. RewardSimulator               : data_spec §5 の適合構造に基づく擬似報酬
                                     (実運用では実クリック/CVに置き換える)
  4. オンライン評価ループ           : purchaseログを順に流して学習・絞り込み
"""

import csv
import math
import random
from collections import defaultdict

import numpy as np

random.seed(0)
np.random.seed(0)

# ============================================================
# 0. データ読み込み
# ============================================================
DATA_DIR = "/home/claude"

def load_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

products = load_csv(f"{DATA_DIR}/products.csv")
ads = load_csv(f"{DATA_DIR}/ads.csv")
access_log = load_csv(f"{DATA_DIR}/access_log.csv")

# area_type 参照(商品マスタから area -> area_type)
AREA_TYPE = {p["area"]: p["area_type"] for p in products}

CAR_TYPES = ["SUV", "minivan", "hatchback"]
AREAS = ["chiba", "funabashi", "kashiwa", "narita", "katsuura"]
AD_CATEGORIES = ["spot", "restaurant", "goods"]

# purchaseイベントのみ抽出(決済後画面の文脈)
purchases = [r for r in access_log if r["event_type"] == "purchase"]

# ============================================================
# 1. 文脈ベクトル x(context, ad) の構築
# ============================================================
# disjoint LinUCB では「腕ごとに独立した係数」を持つが、
# 腕をIDではなく特徴で表すことで新規広告(コールドスタート)にも対応する。
# ここでは1本の共有特徴空間 x(context, ad) を作り、腕は ad とする。
#
# 特徴設計(すべて0/1または正規化済み連続値):
#   [A] 決済文脈の素性     : car_type(3) + area(5) + area_type(2) + price帯(3) + days正規化(1)
#   [B] 広告の素性         : ad_category(3) + is_new(1)
#   [C] 交差(効きどころ)   : area一致(1) + resort×spot/rest(1) + car×goods適合(1)
#                            + car_type×ad_category(9) + area×ad_category(15)

def onehot(value, categories):
    return [1.0 if value == c else 0.0 for c in categories]

PRICE_BINS = [15000, 35000]  # 低 / 中 / 高 の境界(円)
def price_bucket(price):
    p = float(price)
    if p < PRICE_BINS[0]:
        return [1.0, 0.0, 0.0]
    elif p < PRICE_BINS[1]:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]

# goods の車種適合用: target_car_type を見る
def build_context(purchase, ad):
    ct = purchase["car_type"]
    ar = purchase["area"]
    at = AREA_TYPE[ar]
    days = float(purchase["rental_days"])
    price = float(purchase["price"])

    cat = ad["ad_category"]
    is_new = 1.0 if ad["is_new"] == "True" else 0.0
    ad_area = ad["target_area"]          # "" or area
    ad_car = ad["target_car_type"]       # "" or car_type

    # [A] 決済文脈
    feat = []
    feat += onehot(ct, CAR_TYPES)        # 3
    feat += onehot(ar, AREAS)            # 5
    feat += onehot(at, ["urban", "resort"])  # 2
    feat += price_bucket(price)          # 3
    feat += [min(days, 5.0) / 5.0]       # 1 (0..1正規化)

    # [B] 広告素性
    feat += onehot(cat, AD_CATEGORIES)   # 3
    feat += [is_new]                     # 1

    # [C] 交差
    area_match = 1.0 if (ad_area != "" and ad_area == ar) else 0.0
    resort_local = 1.0 if (at == "resort" and cat in ("spot", "restaurant")) else 0.0
    car_goods_fit = 1.0 if (cat == "goods" and ad_car != "" and ad_car == ct) else 0.0
    feat += [area_match, resort_local, car_goods_fit]  # 3

    # car_type × ad_category (3x3=9)
    for c in CAR_TYPES:
        for k in AD_CATEGORIES:
            feat.append(1.0 if (ct == c and cat == k) else 0.0)
    # area × ad_category (5x3=15)
    for a in AREAS:
        for k in AD_CATEGORIES:
            feat.append(1.0 if (ar == a and cat == k) else 0.0)

    return np.array(feat, dtype=float)

# 次元数の確認用に1本作る
_dim = build_context(purchases[0], ads[0]).shape[0]

# ============================================================
# 2. LinUCB(disjoint, 共有特徴空間版)
# ============================================================
# 1つの共有重み theta を、リッジ回帰のオンライン更新で推定する。
# A = D^T D + λI, b = D^T r ; theta = A^{-1} b
# UCB: score = theta^T x + alpha * sqrt(x^T A^{-1} x)
class LinUCBPolicy:
    def __init__(self, dim, alpha=0.5, l2=1.0):
        self.dim = dim
        self.alpha = alpha
        self.A = np.identity(dim) * l2
        self.A_inv = np.identity(dim) / l2
        self.b = np.zeros(dim)
        self.theta = np.zeros(dim)

    def _ucb(self, x):
        mean = float(self.theta @ x)
        var = float(x @ self.A_inv @ x)
        return mean + self.alpha * math.sqrt(max(var, 0.0))

    def rank(self, context_feats):
        # context_feats: list of (ad_id, x)
        scored = [(ad_id, self._ucb(x)) for ad_id, x in context_feats]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def update(self, x, reward):
        self.A += np.outer(x, x)
        self.b += reward * x
        # Sherman-Morrison で A_inv を逐次更新(O(d^2))
        Ax = self.A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self.A_inv -= np.outer(Ax, Ax) / denom
        self.theta = self.A_inv @ self.b

# ============================================================
# 3. 報酬シミュレータ(data_spec §5 の適合構造)
#    実運用では実クリック/CVログに置き換える。
#    ここでは「適合度 -> クリック確率」のロジスティックでベルヌーイ報酬を生成。
# ============================================================
class RewardSimulator:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def click_prob(self, purchase, ad):
        ct = purchase["car_type"]
        ar = purchase["area"]
        at = AREA_TYPE[ar]
        cat = ad["ad_category"]
        ad_area = ad["target_area"]
        ad_car = ad["target_car_type"]

        score = -2.2  # ベース(低めのCTR)
        # 地域一致
        if ad_area != "" and ad_area == ar:
            score += 1.6
        # 観光地では spot/restaurant が効く
        if at == "resort" and cat in ("spot", "restaurant"):
            score += 1.0
        # 都市型では goods がやや効く
        if at == "urban" and cat == "goods":
            score += 0.4
        # 車種×goods 適合
        if cat == "goods" and ad_car != "" and ad_car == ct:
            score += 1.1
        # 地域非依存goodsの薄い汎用効果
        if cat == "goods" and ad_car == "":
            score += 0.2
        # 地域不一致の spot/restaurant は逆効果(他地域の店は出しても刺さらない)
        if ad_area != "" and ad_area != ar:
            score -= 1.2

        return 1.0 / (1.0 + math.exp(-score))

    def sample(self, purchase, ad):
        p = self.click_prob(purchase, ad)
        return 1.0 if self.rng.random() < p else 0.0, p

# ============================================================
# 4. オンライン評価ループ
# ============================================================
TOP_K = 5            # 絞り込み件数
ALPHA = 0.6          # 探索強度
policy = LinUCBPolicy(dim=_dim, alpha=ALPHA, l2=1.0)
sim = RewardSimulator(seed=123)

# 各purchaseで広告全件の文脈を作り、上位Kを絞り込み、
# 絞り込んだK件に対して報酬を観測して学習する(セミバンディット型)。
# 評価指標:
#   - reward@K   : 絞り込んだK件で得られたクリック合計 / ステップ
#   - oracle@K   : 真のclick_probで選んだ上位Kの期待値(上界の目安)
#   - regret     : oracle期待値 - 実現rewardの累積

log_reward_at_k = []
log_oracle_at_k = []
cum_regret = 0.0
regret_curve = []

# 学習推移を見るため、一定間隔でスナップショット
for step, purchase in enumerate(purchases, 1):
    # 全広告の文脈ベクトル
    cf = [(ad["ad_id"], build_context(purchase, ad)) for ad in ads]
    ad_by_id = {ad["ad_id"]: ad for ad in ads}

    # ポリシーで上位Kを絞り込み
    ranked = policy.rank(cf)
    topk = ranked[:TOP_K]

    # oracle(真の確率の上位K)
    oracle_probs = sorted(
        (sim.click_prob(purchase, ad) for ad in ads), reverse=True
    )[:TOP_K]
    oracle_val = sum(oracle_probs)

    # 絞り込んだK件で報酬観測 & 学習
    realized = 0.0
    x_map = {ad_id: x for ad_id, x in cf}
    for ad_id, _ucb_score in topk:
        ad = ad_by_id[ad_id]
        r, _p = sim.sample(purchase, ad)
        realized += r
        policy.update(x_map[ad_id], r)

    log_reward_at_k.append(realized)
    log_oracle_at_k.append(oracle_val)
    cum_regret += (oracle_val - realized)
    regret_curve.append(cum_regret)

# ============================================================
# 5. 結果サマリ
# ============================================================
def moving_avg(xs, w):
    out = []
    s = 0.0
    from collections import deque
    q = deque()
    for v in xs:
        q.append(v); s += v
        if len(q) > w:
            s -= q.popleft()
        out.append(s / len(q))
    return out

n = len(purchases)
print("=== Linear Contextual Bandit (LinUCB) 広告絞り込み ===")
print(f"context次元 d      : {_dim}")
print(f"広告数(腕)         : {len(ads)}  (new={sum(1 for a in ads if a['is_new']=='True')})")
print(f"purchaseステップ数 : {n}")
print(f"絞り込み件数 K     : {TOP_K},  探索強度 alpha={ALPHA}")
print()

first = slice(0, 100)
last = slice(-100, None)
print(f"reward@K  最初の100step平均 : {np.mean(log_reward_at_k[first]):.3f}")
print(f"reward@K  最後の100step平均 : {np.mean(log_reward_at_k[last]):.3f}")
print(f"oracle@K  平均(上界目安)    : {np.mean(log_oracle_at_k):.3f}")
print(f"累積regret(最終)            : {cum_regret:.1f}")
print(f"平均regret/step             : {cum_regret/n:.3f}")

# 学習後、代表的な決済文脈での絞り込み結果を表示
print("\n=== 学習後の絞り込み例 ===")
def show_example(ct, ar, days, price):
    pseudo = {"car_type": ct, "area": ar, "rental_days": str(days),
              "price": str(price)}
    cf = [(ad["ad_id"], build_context(pseudo, ad)) for ad in ads]
    ad_by_id = {ad["ad_id"]: ad for ad in ads}
    ranked = policy.rank(cf)[:TOP_K]
    at = AREA_TYPE[ar]
    print(f"\n[文脈] car={ct}, area={ar}({at}), days={days}, price={price}")
    for rank_i, (ad_id, score) in enumerate(ranked, 1):
        ad = ad_by_id[ad_id]
        tp = sim.click_prob(pseudo, ad)
        tag = " (NEW)" if ad["is_new"] == "True" else ""
        print(f"  {rank_i}. {ad_id} [{ad['ad_category']:10s}] "
              f"{ad['title']}{tag}  UCB={score:.3f} trueCTR={tp:.3f}")

show_example("SUV", "katsuura", 4, 38000)   # 観光地×SUV
show_example("minivan", "chiba", 2, 24000)  # 都市型×minivan
show_example("hatchback", "narita", 3, 20000)  # 観光地×hatchback

# regret曲線をCSVに出力(可視化用)
with open(f"{DATA_DIR}/bandit_regret.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["step", "reward_at_k", "oracle_at_k", "cum_regret",
                "reward_ma50"])
    ma = moving_avg(log_reward_at_k, 50)
    for i in range(n):
        w.writerow([i + 1, log_reward_at_k[i], round(log_oracle_at_k[i], 4),
                    round(regret_curve[i], 4), round(ma[i], 4)])
print("\nregret曲線を bandit_regret.csv に出力しました。")
