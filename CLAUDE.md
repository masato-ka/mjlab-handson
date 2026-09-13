# CLAUDE.md — mjlab PPOハンズオン プロジェクト

このファイルは、Claude(chat)とのセッションで作り込んだ「mjlabでPPO強化学習を学ぶハンズオン」の作業コンテキストを、Claude Codeでの続き作業に引き継ぐためのものです。

## プロジェクト概要

CartPole(swing-up)タスクを題材に、mjlab(MuJoCo-Warpベースの強化学習フレームワーク、Isaac Lab風のmanager-based API)でPPOを学ぶハンズオン教材を2つ作っています。

1. **Craftドキュメント**「mjlabでPPO強化学習するハンズオン」— 理論・解説・リファレンス実装を載せる読み物
2. **Google Colab Notebook**「mjlab_cartpole_handson.ipynb」— 実際に手を動かして実装・学習するハンズオン本体

**設計方針**: 理論と正解実装はCraftドキュメント側に置き、Notebook側は「問題のある実装から出発してユーザー自身が直す」演習に徹する、という役割分担にしている。Notebookには意図的に解答を載せていない。

## 現在のファイルの所在

- Notebook本体(`mjlab_cartpole_handson.ipynb`)は、このチャットセッションのサンドボックス上(`/mnt/user-data/outputs/`)で作成・編集してきたもので、**Claude Codeからは直接アクセスできない**。ユーザーがChat UIからダウンロードして、作業リポジトリに配置する必要がある。
- Craftドキュメントは Craft(craft.do)のワークスペース上にあり、ページ名は「mjlabでPPO強化学習するハンズオン」(親ページ「mjlabのお勉強」の子ページ)。同じ空間に以下の詳細リファレンスページも存在する: Observation, Events, Rewardエンティティ, Entity, Scene, Actuator, Sensor Entity, Term configuration。Claude Codeからは直接読めないので、必要な内容はユーザーにコピーしてもらうか、chat側のCraft MCPコネクタ経由で作業する。

## Notebookの構成(現状)

```
0. セットアップ (nvidia-smi, pip install mjlab / rsl-rl-lib / wandb, wandb.login())

1. Baseline実装
   - MJCF (cartpole.xml) の説明 + writefile
   - Entity/Actuatorの設定(spec_fn, EntityCfg, XmlActuatorCfg) ※これは「与えられたコード」、演習ではない
   - Observationの実装 ※演習(問題のある実装として生角度観測を提示 → cos/sinに直すTODO。解答は載せていない)
   - Actionsの実装   ※演習(TODOのみ。キー名は "effort" にする指示あり)
   - Eventsの実装    ※演習(TODOのみ)
   - Rewardsの実装   ※演習(問題のある実装として一様な生存ボーナスを提示 → 密な報酬に直すTODO。解答は載せていない)
   - Terminationsの実装 ※演習(TODOのみ)
   - 組み立て(cartpole_env_cfg関数) — 上の5つの変数(observations/actions/events/rewards/terminations)を使う。
     演習が未完成(raise NotImplementedError / None のまま)だとここでエラーになるのが仕様。
   - PPOの設定(cartpole_ppo_runner_cfg) + run_training ヘルパー + 学習実行(run_name="baseline")
   - 「参考: 意図的に不適切な実装に戻して比較する」の推奨文(報酬/観測/entropy_coefをそれぞれ元の問題のある状態に戻して再学習し、WandBで比較することを勧める文章。コードは提供していない)
   - GPUマシンがない場合は5章(ONNX+CPU)へ、という誘導リンク

2. ドメインランダマイゼーション
   - mjlab.envs.mdp.dr モジュール(dr.pseudo_inertia, dr.joint_damping)でポールの質量・慣性・関節減衰をランダム化
   - run_name="tier2a_domain_rand"

3. コマンド入力の実装
   - CommandTerm/CommandTermCfg(build(env)メソッド必須の現行API)でカートの目標位置をコマンド化
   - 学習時はランダムサンプリング、play時はset_external_target()で外部入力に差し替え可能
   - reward の centered 項をコマンドの目標値基準に変更(smooth_rewardからの最小差分)
   - run_name="tier3_command_conditioned"

4. 結果の比較と学習済みモデルの実行
   - WandBでの3run比較(baseline / tier2a_domain_rand / tier3_command_conditioned)
   - Tier3のコマンド追従デモ動画(mediapy, オフスクリーン)
   - ローカルでの対話的確認(play_local.py) — なぜColabでは対話的操作ができないかの説明つき
     - play_local.py は PEP 723 インラインスクリプトメタデータ(dependencies: mjlab, rsl-rl-lib)付き
     - 実行は `uv run play_local.py --checkpoint <ckpt> --task {baseline|tier2a_domain_rand|tier3}`
     - ←/→キーでtier3の目標位置操作、Ctrl+右クリックドラッグで外乱

5. ONNXエクスポートとCPUのみでのローカル実行
   - pollen-robotics/microduck_rl の export.py / infer_policy.py を参考にした設計
   - runner.export_policy_to_onnx() でbaseline/tier3をONNX化 + メタデータ埋め込み
   - infer_local.py: mujoco + onnxruntime のみ(PyTorch/mjlab/GPU不要)でCPU上に実行
     - PEP 723 インラインメタデータ(dependencies: mujoco, onnxruntime, numpy)
     - 実行は `uv run infer_local.py --onnx <onnx> --task {baseline|tier3}`
     - mujoco.viewer.launch_passive の key_callback で tier3 の目標位置をリアルタイム操作
     - macOSは `mjpython` が必要な点に注意(uv run --with ... mjpython ...)

まとめ
```

## 守ってほしい設計・執筆方針

- **正解実装をNotebookに書かない**: Observation/Actions/Events/Rewards/Terminationsの5演習はTODO(`raise NotImplementedError` または `= None`)のままにし、「詳細はCraftドキュメント参照」という案内文だけを添える。これは明示的な方針なので、今後の編集でも解答を書き足さないこと。
- **見出しを問いかけ形にしない**: セクション名・見出しは常に「結論」を書く。NG例: 「何が問題か」。OK例: 「生角度観測は状態に対して一意にならない」。これはこのプロジェクトに限らずユーザーの全体方針(Claudeのメモリーにも`/preferences.md`として保存済み)。
- **ローカル実行はすべてuv**: `play_local.py`・`infer_local.py`ともにPEP 723のインラインスクリプトメタデータで依存を宣言し、`uv run <script>.py ...`で実行する。uv自体のインストールは公式サイト(https://docs.astral.sh/uv/)へのリンクのみで済ませ、手順の詳細は書かない。
- **用語の正確さ**: 「生存ボーナスのみの報酬」は疎(sparse)ではなく**密だが一様(uninformative)な報酬**である、という区別を正確に扱うこと(過去に誤って「疎な報酬」と書いて訂正した経緯がある)。
- **actionsディクショナリのキー名**: `"effort"`で統一。
- **スコープ**: Tier1のStep0〜3方式(段階的な悪い状態からの改善)、Tier2-B(外乱イベント)、entropy比較実験は現行版では削除済み。理論的な背景(On-policy/Off-policy、PPOのクリップ更新、GAE、分散環境での学習の工夫など)はCraftドキュメント側に寄せてあるので、Notebookに理論の逐語的な再解説を追加しないこと。

## 技術的な参照・裏取りした情報

- mjlab公式ドキュメント: https://mujocolab.github.io/mjlab/ 、GitHub: https://github.com/mujocolab/mjlab
- mjlabはv1.1.0以降、依存込み(MuJoCo-Warp含む)でPyPI公開(`pip install mjlab`のみでOK)
- `mjlab.envs.mdp.dr`モジュールの型付きドメインランダム化API(`dr.pseudo_inertia`, `dr.joint_damping`, `dr.geom_friction`など。operation/distributionを指定する設計)
- `mjlab.managers.command_manager.CommandTerm` は最近のバージョンで破壊的変更あり:
  - `_update_command(self, env_ids)` に env_ids 引数が必須になった
  - `CommandTermCfg`は`class_type`ではなく`build(self, env)`メソッドを実装する現行パターンに変わった
- `mjlab.envs.mdp.events.push_by_setting_velocity`はフローティングベース(自由継手を持つ)ロボット向けで、CartPoleのような固定ベース+関節駆動のエンティティには直接使えない(→ Tier2-Bで関節速度に直接インパルスを与えるカスタムイベントを実装した経緯あり。現行版ではTier2-B自体を削除済み)
- `mjlab.viewer.NativeMujocoViewer`は`key_callback`引数を持つ(GLFUキーコード: 262=Right, 263=Left)。`tick()`ベースのループの正確な回し方はバージョン差が出やすいので要注意
- `mujoco.viewer.launch_passive(model, data, key_callback=...)`は標準mujocoパッケージのAPIで、こちらは安定している。ただしmacOSでは`mjpython`が必要
- 参考にした実プロジェクト: [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) の `src/mjlab_microduck/export.py`(ONNXエクスポートの正しい経路: `runner.export_policy_to_onnx()`を必ず使う)と `scripts/infer_policy.py`(学習はGPU、デプロイはCPU+ONNXという設計思想)

## 次にやりそうなこと(推測。ユーザーに確認してから進めること)

- Notebookをリポジトリに配置し、実際にColabで通しで実行して動作確認
- Craftドキュメントとの記述の整合性チェック(特に用語やコードスニペットの同期)
- 演習セル(Observation/Actions/Events/Rewards/Terminations)を自分で実装してみて、意図通りに学習が進むか検証
- 必要であれば`play_local.py`/`infer_local.py`をリポジトリ内のスクリプトとして独立させる(現状はNotebookの`%%writefile`セルの中にコードが埋まっている)
