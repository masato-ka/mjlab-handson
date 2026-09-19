# mjlab PPOハンズオン

CartPole(swing-up)タスクを題材に、[mjlab](https://github.com/mujocolab/mjlab)(MuJoCo-Warpベースの強化学習フレームワーク、Isaac Lab風のmanager-based API)でPPOを学ぶハンズオン教材です。

## 1. Overview

このハンズオンは2つの教材で構成されています。

- **Craftドキュメント「mjlabでPPO強化学習するハンズオン」** — 理論の解説と、各演習の正解実装を載せたリファレンス。リンク: **準備中**(公開後にここに追記します)
- **`mjlab_cartpole_handson.ipynb`**(このリポジトリ本体) — Google Colab上で実際に手を動かして実装・学習するNotebook

役割分担として、理論と正解実装はCraftドキュメント側に、Notebook側は「問題のある実装から出発してユーザー自身が直す」演習に徹しています。Observation/Actions/Events/Rewards/Terminationsの5演習、および末尾の追加課題(風車課題)には解答を載せていません。詰まった場合はCraftドキュメントの該当セクションを参照してください。

ハンズオンは基本的にGoogle Colab上での実行を前提としています。ローカル環境の準備は、学習済みモデルをローカルで動かして確認したい場合(対話操作・CPUのみでの推論など)に必要です。

## 2. リポジトリの説明

| ファイル | 役割 |
|---|---|
| `mjlab_cartpole_handson.ipynb` | ハンズオン本体のNotebook。Google Colabで開いて使う |
| `cartpole.xml` | CartPoleのMJCF(物理構造)定義。ローカルスクリプトから参照される |
| `play_local.py` | 学習済みモデルをローカルのGPU上でmjlab経由で対話実行するスクリプト |
| `infer_local.py` | 学習済みモデルをONNX化した上で、CPUのみ(mjlab/PyTorch/GPU不要)で実行するスクリプト |
| `pyproject.toml` / `uv.lock` | ローカル実行用の依存関係定義(uv管理) |

Notebookは以下の章構成になっています。

1. **Baseline実装** — MJCF定義・Entity/Actuator設定、Observation/Actions/Events/Rewards/Terminationsの5演習、PPO学習の実行(`run_name="baseline"`)。学習直後にColab上での動画デモと、ローカル実行(`play_local.py`/`infer_local.py`)の手順も用意しています
2. **ドメインランダマイゼーション** — `mjlab.envs.mdp.dr`でポールの質量・慣性・関節減衰をランダム化して学習(`run_name="tier2a_domain_rand"`)
3. **コマンド入力** — `CommandTerm`でカートの目標位置をコマンド化し、追従報酬を学習(`run_name="tier3_command_conditioned"`)
4. **結果の比較と学習済みモデルの実行** — WandBでの3run比較、コマンド追従デモ動画
5. **ONNXエクスポートとCPUのみでのローカル実行** — 各Tierのローカル実行方法のまとめ

最後に、報酬をゼロから自分で設計する追加課題(風車課題)を用意しています。

## 3. インストール方法

ローカル環境の準備は、学習済みモデルをローカルで動かしたい場合にのみ必要です(Colab上での実装・学習だけを行う場合は不要です)。

### 3.1 uvのインストール

このリポジトリのローカル実行はすべて[uv](https://docs.astral.sh/uv/)を使います。インストール手順は公式サイトを参照してください: https://docs.astral.sh/uv/getting-started/installation/

### 3.2 依存関係のインストール

依存関係はGPUの有無で2系統に分かれています。

```bash
# CPUのみの依存関係(mujoco, onnxruntime, numpy)をインストール
# GPUを持たない環境ではこれだけでよい
uv sync
```

```bash
# NVIDIA GPU(CUDA)を搭載したマシンでのみ実行すること
# play_local.pyでの対話実行に必要
uv sync --extra gpu
```

`gpu` extraには`mjlab`/`rsl-rl-lib`/`wandb`/`mediapy`が含まれます。mjlabはGPU(CUDA)前提のフレームワークのため、GPUを持たないマシンでこのextraをインストールしても正しく動作しません。

学習(PPOの実行)自体は常にColab上のGPUで行う設計です。ローカルにGPUがない場合でも、Colabで学習・ONNXエクスポートしたモデルを`infer_local.py`でダウンロードして動かすことができます(`uv sync`だけで準備できます)。

## 4. Google Colaboratoryの準備

Notebook本体(`mjlab_cartpole_handson.ipynb`)は先頭の「Open in Colab」バッジから直接開けます。もしくは以下のURLに直接アクセスしてください。

```
https://colab.research.google.com/github/masato-ka/mjlab-handson/blob/main/mjlab_cartpole_handson.ipynb
```

開いたら、以下を確認・実施してください。

1. **ランタイムをGPUに変更する**:メニューの「ランタイム」→「ランタイムのタイプを変更」→ハードウェアアクセラレータで**GPU(T4など)**を選択する
2. **セットアップセルを実行する**:先頭の「0. セットアップ」セルで`mjlab`・`rsl-rl-lib`・`wandb`・`mediapy`をインストールし、GPUが認識されているか(`torch.cuda.is_available()`)を確認する
3. **WandBにログインする**:セットアップセル内の`wandb.login()`実行時にAPIキーの入力を求められます。WandBアカウントを持っていない場合は匿名実行を選択することもできます

Google Colabのアカウント・GPUクォータについては[Colabの公式ヘルプ](https://colab.research.google.com/)を参照してください。

## 5. ローカルスクリプトの実行

Colab上で学習・ONNXエクスポートしたモデルをダウンロードしたら、ローカルで動かして確認できます。`play_local.py`・`infer_local.py`・`cartpole.xml`はこのリポジトリに直接含まれています。前者2つはPEP 723のインラインスクリプトメタデータに依存関係を宣言した単体スクリプトで、`uv sync`とは独立に、それぞれが必要な依存だけを実行時に自動解決します。

```bash
# 学習済みチェックポイント(.pt)とcartpole.xmlを使った対話実行(要 mjlab, GPU向け)
uv run play_local.py --checkpoint model_499.pt --task baseline

# ONNX化済みモデルによるCPUのみでの対話実行(mjlab/PyTorch/GPU不要)
uv run infer_local.py --onnx baseline_policy.onnx --task baseline
```

操作方法:
- `←` / `→` キー: `tier3`タスクでの目標カート位置の操作
- `Ctrl`+右クリックドラッグ: カート/ポールへの外乱付与(mujocoビューア標準機能)

### macOSでの注意

`mujoco.viewer.launch_passive`はmacOSでは通常の`python`コマンドでは動作せず、mujocoパッケージに同梱されている`mjpython`コマンドを代わりに使う必要があります。

```bash
uv run mjpython infer_local.py --onnx baseline_policy.onnx --task baseline
```

uv管理のPython(uv自身がダウンロードしたCPython)を使っている場合、上のコマンドが`Library not loaded: @executable_path/../lib/libpython3.1x.dylib`のようなdlopenエラーで失敗することがあります。これはmjpython・uv双方の既知の未解決の相性問題です([google-deepmind/mujoco#1923](https://github.com/google-deepmind/mujoco/issues/1923)、[astral-sh/uv#8953](https://github.com/astral-sh/uv/issues/8953))。その場合は`DYLD_FALLBACK_LIBRARY_PATH`でuv管理Pythonのlibディレクトリを明示的に教えてあげると回避できます。

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
uv run mjpython infer_local.py --onnx baseline_policy.onnx --task baseline
```
