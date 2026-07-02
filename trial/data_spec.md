# データスペック：カーレンタル広告レコメンド用 想定データ

## 0. 目的と全体像

決済後画面に広告を出すための Contextual Bandit を想定した、合成データの仕様。
カーレンタル予約サイトのアクセスログを起点に、予約完了（決済）時の文脈で
広告候補を絞り込む、という用途を想定する。

データは3種類。

1. **アクセスログデータ**（`access_log`）：商品閲覧・商品購入のイベント列
2. **商品データ**（`products`）：閲覧・購入される車のマスタ
3. **広告データ**（`ads`）：決済後画面に出す広告のマスタ

---

## 1. 共通の定義域（カテゴリ値）

### 1.1 車種 `car_type`
| 値 | 説明 |
|---|---|
| `SUV` | SUV |
| `minivan` | ミニバン |
| `hatchback` | ハッチバック |

### 1.2 地域 `area`（千葉の主要都市）
| 値 | 都市 |
|---|---|
| `chiba` | 千葉市 |
| `funabashi` | 船橋市 |
| `kashiwa` | 柏市 |
| `narita` | 成田市 |
| `katsuura` | 勝浦市 |

- 内陸/都市型（`chiba`, `funabashi`, `kashiwa`）と、観光/海寄り（`narita`, `katsuura`）の
  性格差を持たせ、広告との適合に効くようにする。

### 1.3 商品（車）= `car_type` × `area` の組み合わせ
- 商品は「車種 × 地域」の直積で定義する（3 × 5 = 15 商品）。
- 1予約は、この15通りから1つを選んで決済したものとする。

---

## 2. アクセスログデータ `access_log`

1ユーザーの1セッションを、複数の閲覧イベント＋（任意で）1つの購入イベントの
列として表現する。閲覧は車種と地域を回遊し、購入は最終的に1つの組み合わせを選ぶ。

### 2.1 スキーマ
| カラム | 型 | 説明 |
|---|---|---|
| `event_id` | string | イベント一意ID（`evt_` + 連番） |
| `session_id` | string | セッションID（`sess_` + 連番） |
| `user_id` | string | 仮想ユーザーID（`user_` + 連番） |
| `event_type` | enum | `view`（商品閲覧） / `purchase`（商品購入＝決済） |
| `event_seq` | int | セッション内のイベント順序（1始まり） |
| `timestamp` | datetime | イベント発生時刻（ISO 8601） |
| `car_type` | enum | 閲覧/購入した車種（§1.1） |
| `area` | enum | 閲覧/購入した地域（§1.2） |
| `product_id` | string | 対象商品ID（§3 の `product_id`、`car_type`×`area`に対応） |
| `rental_days` | int | レンタル日数（purchase時のみ非null。view時はnull） |
| `price` | int | 決済金額（円, purchase時のみ非null。view時はnull） |
| `device` | enum | `mobile` / `desktop` |

### 2.2 セッションの生成ルール
- 1セッションは数件の `view`（回遊）を持ち、確率的に末尾へ `purchase` を1件付ける
  （購入に至らないセッションも一定割合で存在させる）。
- `view` の `car_type` / `area` はセッションごとに一定の嗜好（後述の嗜好ベクトル）に
  従ってサンプリングし、回遊感を出す。
- `purchase` の `car_type` / `area` は、そのセッションで閲覧した中から
  嗜好重みに比例して1つ選ぶ。
- `event_seq` はセッション内で連番。`timestamp` はセッション内で単調増加。

### 2.3 価格レンジ（purchase時）
| 車種 | 1日あたり基準価格（円） |
|---|---|
| `SUV` | 9,000 |
| `minivan` | 11,000 |
| `hatchback` | 6,000 |

- `price = 基準価格 × rental_days × area係数 × ノイズ`
- `rental_days` は 1〜5 を想定（観光地ほど長め寄りにする）。
- `area係数`：観光地（`narita`, `katsuura`）はやや高め（1.05〜1.1）。

---

## 3. 商品データ `products`

`car_type` × `area` の15通りを定義するマスタ。

### 3.1 スキーマ
| カラム | 型 | 説明 |
|---|---|---|
| `product_id` | string | 商品ID（`prod_` + 連番、`car_type`×`area`で一意） |
| `car_type` | enum | 車種（§1.1） |
| `area` | enum | 地域（§1.2） |
| `seats` | int | 乗車定員（SUV=5, minivan=8, hatchback=5 を基準） |
| `base_price_per_day` | int | 1日あたり基準価格（§2.3） |
| `area_type` | enum | `urban`（都市型） / `resort`（観光型） |

---

## 4. 広告データ `ads`

決済後画面に出す広告のマスタ。各地域のおすすめスポット・レストラン、
または旅行準備におすすめの商品。bandit の「腕（arm）」に相当する。

### 4.1 広告カテゴリ `ad_category`
| 値 | 説明 | 地域依存 |
|---|---|---|
| `spot` | おすすめ観光スポット | あり（特定areaに紐づく） |
| `restaurant` | おすすめレストラン | あり（特定areaに紐づく） |
| `goods` | 旅行準備グッズ | なし（area非依存、全国共通） |

### 4.2 スキーマ
| カラム | 型 | 説明 |
|---|---|---|
| `ad_id` | string | 広告ID（`ad_` + 連番） |
| `ad_category` | enum | `spot` / `restaurant` / `goods`（§4.1） |
| `title` | string | 広告タイトル（例：「成田山表参道グルメ」） |
| `target_area` | enum/null | 紐づく地域（§1.2）。`goods` はnull（全地域対象） |
| `target_car_type` | enum/null | 想定車種（任意）。例：アウトドアグッズ→`SUV`寄り。無指定はnull |
| `advertiser` | string | 広告主名（合成） |
| `is_new` | bool | 新規広告フラグ（コールドスタート検証用。一部をtrueにする） |

### 4.3 広告の構成方針
- `spot` と `restaurant` は、5地域それぞれに複数件用意する（area紐づけ）。
- `goods` は area非依存で、車種寄り（`SUV`→アウトドア、`minivan`→ファミリー等）を
  `target_car_type` で表現する。
- `is_new=true` の広告を数件混ぜ、過去ログを持たない腕としてコールドスタート検証に使う。

---

## 5. 広告適合の想定（報酬設計の伏線）

実データ生成では報酬（クリック等）は付けないが、後段の bandit で報酬を作りやすいよう、
以下の「効きやすさ」の構造を意図して埋め込む。

- **地域一致**：`purchase.area == ad.target_area` の広告は適合が高い。
- **観光地バイアス**：`area_type == resort`（`narita`, `katsuura`）では `spot` / `restaurant` が効きやすい。
- **車種と goods**：`SUV` 購入者にはアウトドア系 `goods` が効きやすい、`minivan` にはファミリー系。
- **新規広告**：`is_new=true` は履歴がないため、探索を通じてのみ評価される想定。

---

## 6. 出力ファイル

| ファイル名 | 内容 |
|---|---|
| `products.csv` | §3 の商品マスタ（15行） |
| `ads.csv` | §4 の広告マスタ |
| `access_log.csv` | §2 のアクセスログ（複数セッション分のイベント列） |

- 文字コード UTF-8、ヘッダ付きCSV。
- 乱数シードを固定し、再現可能にする。
