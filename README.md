# mjlab PPOハンズオン

CartPole(swing-up)タスクを題材に、[mjlab](https://github.com/mujocolab/mjlab)(MuJoCo-Warpベースの強化学習フレームワーク、Isaac Lab風のmanager-based API)でPPOを学ぶハンズオン教材。

## 構成は2つの教材に分かれている

- **Craftドキュメント**「mjlabでPPO強化学習するハンズオン」— 理論・解説・リファレンス実装(Craftワークスペース上、このリポジトリには含まれない)
- **`mjlab_cartpole_handson.ipynb`**(本体)— Google Colab上で実際に手を動かして実装・学習するノートブック

理論と正解実装はCraftドキュメント側、Notebook側は「問題のある実装から出発してユーザー自身が直す」演習に徹する、という役割分担になっている。Observation/Actions/Events/Rewards/Terminationsの5演習には解答を載せていない。

## ノートブックは5段階で構成されている

1. **Baseline実装** — MJCF定義・Entity/Actuator設定・5つの演習・PPO学習(`run_name="baseline"`)
2. **ドメインランダマイゼーション** — `mjlab.envs.mdp.dr`でポールの質量・慣性・関節減衰をランダム化(`run_name="tier2a_domain_rand"`)
3. **コマンド入力** — `CommandTerm`でカートの目標位置をコマンド化し、追従報酬を学習(`run_name="tier3_command_conditioned"`)
4. **結果の比較と学習済みモデルの実行** — WandBでの3run比較、コマンド追従デモ動画、`play_local.py`によるローカル対話実行
5. **ONNXエクスポートとCPUのみでのローカル実行** — `runner.export_policy_to_onnx()`でONNX化し、`infer_local.py`でPyTorch/mjlab/GPUなしに実行

NVIDIA GPUを持たない環境では、1〜4章(学習)は実行できないため、ONNX化済みの学習済みモデルを使う5章から始めるとよい。

## セットアップはuvで行う

依存関係は用途別に2系統に分かれている。

```bash
# CPUのみの依存関係(mujoco, onnxruntime, numpy)をインストール
# GPUを持たない環境ではこれだけでよい
uv sync
```

```bash
# NVIDIA GPU(CUDA)を搭載したマシンでのみ実行すること
# ノートブックの1〜4章(mjlabでのPPO学習)とplay_local.pyに必要
uv sync --extra gpu
```

`gpu` extraには `mjlab` / `rsl-rl-lib` / `wandb` / `mediapy` が含まれる。mjlabはGPU(CUDA)前提のフレームワークのため、GPUを持たないマシンでこのextraをインストールしても正しく動作しない。

uv自体のインストール手順は公式サイトを参照: https://docs.astral.sh/uv/

## ローカルスクリプトはuv runで実行する

`play_local.py`・`infer_local.py`・`cartpole.xml`はこのリポジトリに直接含まれている。前者2つはPEP 723のインラインスクリプトメタデータに依存関係を宣言した単体スクリプトで、`uv sync`とは独立に、それぞれが必要な依存だけを実行時に自動解決する。

```bash
# 学習済みチェックポイント(.pt)とcartpole.xmlを使った対話実行(要 mjlab, GPU向け)
uv run play_local.py --checkpoint model_499.pt --task baseline

# ONNX化済みモデルによるCPUのみでの対話実行(mjlab/PyTorch/GPU不要)
uv run infer_local.py --onnx baseline_policy.onnx --task baseline
```

macOSで`infer_local.py`のビューアを使う場合は`mjpython`が必要:

```bash
uv run --with mujoco --with onnxruntime --with numpy mjpython infer_local.py --onnx baseline_policy.onnx --task baseline
```

操作方法:
- `←` / `→` キー: `tier3`タスクでの目標カート位置の操作
- `Ctrl`+右クリックドラッグ: カート/ポールへの外乱付与(mujocoビューア標準機能)

## 用語

- 「生存ボーナスのみの報酬」は疎(sparse)な報酬ではなく、**密だが一様(uninformative)な報酬**である。
