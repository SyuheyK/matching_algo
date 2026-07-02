# Contextual Bandit による広告絞り込み設計ドキュメント

このディレクトリには、決済後広告面における広告絞り込み問題を Contextual Bandit として定式化した調査・設計ドキュメントを配置しています。

## ファイル

- `contextual_bandit_formulation.tex`
  - 問題設定、報酬設計、特徴量設計、フィードバックバイアス、候補アルゴリズム比較、実装計画、参考文献をまとめた TeX ソースです。
- `contextual_bandit_formulation.pdf`
  - 上記 TeX の内容を確認しやすいように PDF 化した成果物です。

## このドキュメントで扱う主な論点

- 決済文脈と広告特徴を使った top-K 広告絞り込みの定式化
- click / conversion / cost-aware reward の整理
- LinUCB、Hybrid LinUCB、Linear Thompson Sampling、Logistic/GLM Bandit、Neural Bandit の比較
- RTB における落札後にしか観測できない報酬、遅延報酬、selection bias への注意点
- propensity logging、IPS / SNIPS / doubly robust evaluation などのオフライン評価方針
- まず実装すべき shared LinUCB / LinTS baseline から hybrid / GLM へ進む段階的な実装計画
