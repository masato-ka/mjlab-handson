#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mujoco",
#     "onnxruntime",
#     "numpy",
# ]
# ///
"""
infer_local.py
================
学習済みcartpole方策をONNX(onnxruntime)で読み込み、通常のmujocoパッケージだけで
CPU上で動かすスクリプト。mjlab / PyTorch / GPUは一切不要。

pollen-robotics/microduck_rl の export.py / infer_policy.py と同じ考え方
(学習はGPU上のmjlabで行い、デプロイはCPU上のONNX + mujocoで行う)を、
CartPoleの規模に合わせて最小構成にしたもの。

事前準備:
  uvをインストールしておく(公式サイト参照: https://docs.astral.sh/uv/getting-started/installation/)。
  パッケージのインストールは不要。上のインラインスクリプトメタデータ(PEP 723)を
  uv run が読み取り、実行時に自動で一時的な仮想環境を構築する。
  このファイルと同じフォルダに cartpole.xml と *.onnx を置く

実行例:
  uv run infer_local.py --onnx baseline_policy.onnx --task baseline
  uv run infer_local.py --onnx tier3_policy.onnx --task tier3

操作方法(--task tier3 のときのみ):
  ← / → キー            : 目標カート位置を左右に動かす
  Ctrl+右クリックドラッグ : カート/ポールに外力を加える(mujocoビューア標準機能)
  ウィンドウを閉じる      : 終了

macOSの注意:
  mujoco.viewer.launch_passive は通常のpython/uv runでは動作しません。
  mujocoパッケージに同梱されている mjpython コマンドを使う必要があります:

    uv run mjpython infer_local.py --onnx baseline_policy.onnx --task baseline

  uv管理のPython(uv自身がダウンロードしたCPython)を使っている場合、上のコマンドが
    Library not loaded: @executable_path/../lib/libpython3.1x.dylib
  のようなdlopenエラーで失敗することがあります。これはmjpythonがuvの仮想環境
  (.venv/lib/)に libpython*.dylib を探しにいく一方、実体はuvのPython本体側
  (~/.local/share/uv/python/.../lib/)にしかないためで、mjpython・uv双方の既知の
  未解決の相性問題です(google-deepmind/mujoco#1923、astral-sh/uv#8953)。
  その場合は、DYLD_FALLBACK_LIBRARY_PATH でuv管理Pythonのlibディレクトリを
  明示的に教えてあげると回避できます:

    export DYLD_FALLBACK_LIBRARY_PATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
    uv run mjpython infer_local.py --onnx baseline_policy.onnx --task baseline
"""
import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

DECIMATION = 5              # 学習時と同じ制御周期(物理5ステップに1回だけpolicyを呼ぶ)
PHYSICS_TIMESTEP = 0.01     # cartpole.xmlの<option timestep="0.01"/>と一致させる
CTRL_RANGE = (-1.5, 1.5)    # cartpole.xmlのactuator ctrlrangeと一致させる
TARGET_STEP = 0.1
TARGET_RANGE = (-1.2, 1.2)  # CartTargetCommandCfg.target_rangeと一致させる


def get_joint_addrs(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    """関節名からqpos/qvel配列上のインデックスを取得する。"""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint '{joint_name}' が見つかりません")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


class CartpoleObservation:
    """観測ベクトルを組み立てるクラス。

    並び順・内容は、Notebook内のObservationTermCfgの定義と完全に一致させる必要がある。
    ここがズレると、ONNXは正常に動くがロボット(この場合はCartPole)は
    まったく見当違いの入力を渡されて暴走する、という一番気づきにくい不具合になる。
    """

    def __init__(self, model: mujoco.MjModel, task: str):
        self.task = task
        self.cart_qpos, self.cart_qvel = get_joint_addrs(model, "slider")
        self.hinge_qpos, self.hinge_qvel = get_joint_addrs(model, "hinge_1")
        self.target = 0.0  # tier3のみ使用。キー入力で外部から更新される

    def build(self, data: mujoco.MjData) -> np.ndarray:
        cart_pos = float(data.qpos[self.cart_qpos])
        theta = float(data.qpos[self.hinge_qpos])
        cart_vel = float(data.qvel[self.cart_qvel])
        pole_vel = float(data.qvel[self.hinge_qvel])

        if self.task == "tier3":
            # cart_pos, cos(theta), sin(theta), cart_vel, pole_vel, cart_target (6次元)
            obs = [cart_pos, np.cos(theta), np.sin(theta), cart_vel, pole_vel, self.target]
        else:
            # baseline / tier2a_domain_rand (5次元)。ドメインランダム化はイベントのみの変更なので観測形状は同じ
            obs = [cart_pos, np.cos(theta), np.sin(theta), cart_vel, pole_vel]
        return np.asarray(obs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="CartPole ONNX policy inference (CPU-only, no mjlab)")
    parser.add_argument("--onnx", required=True, help="ONNXポリシーファイルのパス")
    parser.add_argument("--xml", default="cartpole.xml", help="MJCFファイルのパス")
    parser.add_argument(
        "--task",
        choices=["baseline", "tier3"],
        default="baseline",
        help="観測の組み立て方(次元)を選ぶ。tier2a_domain_randはbaselineと同じ観測形状なのでbaselineを指定する",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = PHYSICS_TIMESTEP
    # 学習時(SimulationCfg(mujoco=MujocoCfg(..., disableflags=("contact",))))と同じく接触計算を無効化する。
    # cartpole.xmlの見た目用レール(rail1/rail2)はカート本体のgeomと常時重なる位置にあり、
    # contactを有効なままにすると学習時には存在しなかった接触反力が生じ、方策が経験していない
    # 外乱でカートがレール端に押しやられてしまう。
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Loaded ONNX policy: input={session.get_inputs()[0].shape}, output={session.get_outputs()[0].shape}")

    obs_builder = CartpoleObservation(model, args.task)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "slide")

    # swing-up初期状態(ポールを下向きに垂らした状態からスタート)
    data.qpos[obs_builder.hinge_qpos] = np.pi
    mujoco.mj_forward(model, data)

    def key_callback(keycode: int):
        """GLFWキーコード: 262=Right, 263=Left。tier3のときだけ目標カート位置を動かす。"""
        if args.task != "tier3":
            return
        if keycode == 262:
            obs_builder.target = min(obs_builder.target + TARGET_STEP, TARGET_RANGE[1])
            print(f"[key] 目標カート位置: {obs_builder.target:.2f}")
        elif keycode == 263:
            obs_builder.target = max(obs_builder.target - TARGET_STEP, TARGET_RANGE[0])
            print(f"[key] 目標カート位置: {obs_builder.target:.2f}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("ビューアを起動しました。ウィンドウを閉じると終了します。")
        while viewer.is_running():
            step_start = time.time()

            obs = obs_builder.build(data).reshape(1, -1)
            action = session.run([output_name], {input_name: obs})[0]
            ctrl = float(np.clip(action[0, 0], *CTRL_RANGE))

            # 学習時と同じdecimation(物理DECIMATIONステップに1回だけpolicyを呼ぶ)
            data.ctrl[actuator_id] = ctrl
            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            viewer.sync()

            # 実時間に同期させる(ラフなタイムキーピング。ドリフトは許容する)
            elapsed = time.time() - step_start
            sleep_time = (DECIMATION * PHYSICS_TIMESTEP) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
