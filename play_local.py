#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mjlab",
#     "rsl-rl-lib",
# ]
# ///
"""
play_local.py
==============
学習済みcartpole方策を、MuJoCoのネイティブビューアでインタラクティブに実行するスクリプト。
Google Colabでは実行できません(ディスプレイとリアルタイムのキー入力が必要なため)。
ディスプレイのあるローカルPC(NVIDIA GPU、または評価のみならmacOSでも可)で実行してください。

事前準備:
  uvをインストールしておく(公式サイト参照: https://docs.astral.sh/uv/getting-started/installation/)。
  パッケージのインストールは不要。上のインラインスクリプトメタデータ(PEP 723)を
  uv run が読み取り、実行時に自動で一時的な仮想環境を構築する。
  このファイルと同じフォルダに cartpole.xml とチェックポイント(.pt)を置く

実行例:
  uv run play_local.py --checkpoint model_499.pt --task baseline
  uv run play_local.py --checkpoint model_499.pt --task tier2a_domain_rand
  uv run play_local.py --checkpoint model_499.pt --task tier3

操作方法:
  ← / → キー         : (tier3のみ)目標カート位置を左右に動かす
  Ctrl+右クリックドラッグ : カート/ポールに外力を加える(ビューア標準機能。外乱耐性の確認に使える)
  Q / ウィンドウを閉じる : 終了
"""
import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import torch

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg, ManagerBasedRlEnv
from mjlab.envs.mdp import (
    joint_pos_rel, joint_vel_rel, reset_joints_by_offset, time_out,
    JointEffortActionCfg, generated_commands,
)
from mjlab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, SceneEntityCfg, TerminationTermCfg,
)
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import NativeMujocoViewer

_CARTPOLE_XML = Path(__file__).parent / "cartpole.xml"


def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_CARTPOLE_XML))


_ARTICULATION = EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=("slider",)),))
_SWINGUP_INIT = EntityCfg.InitialStateCfg(joint_pos={"slider": 0.0, "hinge_1": math.pi}, joint_vel={".*": 0.0})


def _entity_cfg() -> EntityCfg:
    return EntityCfg(spec_fn=_get_spec, articulation=_ARTICULATION, init_state=_SWINGUP_INIT)


def pole_angle_cos_sin(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def _tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
    return torch.exp(-0.5 * (x / margin) ** 2)


def cartpole_smooth_reward(env, cart_cfg: SceneEntityCfg, hinge_cfg: SceneEntityCfg) -> torch.Tensor:
    cart = env.scene[cart_cfg.name]
    angle = cart.data.joint_pos[:, hinge_cfg.joint_ids]
    cos_angle = torch.cos(angle).squeeze(-1)
    cart_pos = cart.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
    pole_vel = cart.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
    ctrl = env.action_manager.action.squeeze(-1)
    upright = (cos_angle + 1.0) / 2.0
    centered = (1.0 + _tolerance(cart_pos, margin=1.0)) / 2.0
    small_ctrl = (4.0 + _tolerance(ctrl, margin=1.0)) / 5.0
    small_vel = (1.0 + _tolerance(pole_vel, margin=5.0)) / 2.0
    return upright * centered * small_ctrl * small_vel


def baseline_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
    hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
    actor_terms = {
        "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
        "pole_angle": ObservationTermCfg(func=pole_angle_cos_sin, params={"asset_cfg": hinge_cfg}),
        "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
        "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinge_cfg}),
    }
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"cartpole": _entity_cfg()}, num_envs=num_envs, env_spacing=4.0,
        ),
        observations={"actor": ObservationGroupCfg(actor_terms), "critic": ObservationGroupCfg({**actor_terms})},
        actions={"effort": JointEffortActionCfg(entity_name="cartpole", actuator_names=("slider",), scale=1.0)},
        events={
            "reset_slider": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (-0.01, 0.01), "asset_cfg": cart_cfg}),
            "reset_hinge": EventTermCfg(func=reset_joints_by_offset, mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (-0.01, 0.01), "asset_cfg": hinge_cfg}),
        },
        rewards={"smooth_reward": RewardTermCfg(func=cartpole_smooth_reward, weight=1.0,
                params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg})},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",))),
        decimation=5,
        episode_length_s=9999.0,  # play時はタイムアウトなし
    )


class CartTargetCommand(CommandTerm):
    def __init__(self, cfg: "CartTargetCommandCfg", env):
        super().__init__(cfg, env)
        self._target = torch.zeros(env.num_envs, device=env.device)
        self._external_override = None
        self._cart_joint_idx = list(env.scene["cartpole"].joint_names).index("slider")

    @property
    def command(self) -> torch.Tensor:
        return self._target.unsqueeze(-1)

    def _resample_command(self, env_ids: torch.Tensor):
        if self._external_override is not None:
            self._target[env_ids] = self._external_override[env_ids]
        else:
            self._target[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.target_range)

    def _update_command(self, env_ids):
        if self._external_override is not None:
            self._target[:] = self._external_override

    def _update_metrics(self):
        cart = self._env.scene["cartpole"]
        cart_pos = cart.data.joint_pos[:, self._cart_joint_idx]
        self.metrics["cart_target_error"] = (self._target - cart_pos).abs()

    def set_external_target(self, target):
        self._external_override = target


@dataclass(kw_only=True)
class CartTargetCommandCfg(CommandTermCfg):
    entity_name: str = "cartpole"
    resampling_time_range: tuple = (3.0, 6.0)
    target_range: tuple = (-1.2, 1.2)

    def build(self, env) -> CartTargetCommand:
        return CartTargetCommand(self, env)


def cartpole_command_reward(env, cart_cfg: SceneEntityCfg, hinge_cfg: SceneEntityCfg,
                             command_name: str = "cart_target") -> torch.Tensor:
    cart = env.scene[cart_cfg.name]
    angle = cart.data.joint_pos[:, hinge_cfg.joint_ids]
    cos_angle = torch.cos(angle).squeeze(-1)
    cart_pos = cart.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
    pole_vel = cart.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
    ctrl = env.action_manager.action.squeeze(-1)
    target = env.command_manager.get_command(command_name).squeeze(-1)
    upright = (cos_angle + 1.0) / 2.0
    centered = (1.0 + _tolerance(cart_pos - target, margin=1.0)) / 2.0
    small_ctrl = (4.0 + _tolerance(ctrl, margin=1.0)) / 5.0
    small_vel = (1.0 + _tolerance(pole_vel, margin=5.0)) / 2.0
    return upright * centered * small_ctrl * small_vel


def command_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    cfg = baseline_env_cfg(num_envs=num_envs)
    cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
    hinge_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
    cfg.commands = {"cart_target": CartTargetCommandCfg()}
    actor_terms = {
        "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
        "pole_angle": ObservationTermCfg(func=pole_angle_cos_sin, params={"asset_cfg": hinge_cfg}),
        "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
        "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinge_cfg}),
        "cart_target": ObservationTermCfg(func=generated_commands, params={"command_name": "cart_target"}),
    }
    cfg.observations = {"actor": ObservationGroupCfg(actor_terms), "critic": ObservationGroupCfg({**actor_terms})}
    cfg.rewards = {"command_reward": RewardTermCfg(func=cartpole_command_reward, weight=1.0,
                    params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg, "command_name": "cart_target"})}
    return cfg


# ── Tier2-A: ドメインランダム化(観測/行動は共通。イベントも込みにして頑健性を見る) ──
def domain_rand_env_cfg(num_envs: int = 1) -> ManagerBasedRlEnvCfg:
    from mjlab.envs.mdp import dr
    cfg = baseline_env_cfg(num_envs=num_envs)
    pole_cfg = SceneEntityCfg("cartpole", body_names=["pole_1"])
    hinge_j_cfg = SceneEntityCfg("cartpole", joint_names=["hinge_1"])
    slider_j_cfg = SceneEntityCfg("cartpole", joint_names=["slider"])
    cfg.events = dict(cfg.events)
    cfg.events["dr_pole_mass_inertia"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": pole_cfg, "alpha_range": (-0.3, 0.3)},
    )
    cfg.events["dr_hinge_damping"] = EventTermCfg(
        mode="reset", func=dr.joint_damping,
        params={"asset_cfg": hinge_j_cfg, "ranges": (0.5, 3.0), "operation": "scale"},
    )
    cfg.events["dr_slider_damping"] = EventTermCfg(
        mode="reset", func=dr.joint_damping,
        params={"asset_cfg": slider_j_cfg, "ranges": (0.5, 2.0), "operation": "scale"},
    )
    return cfg


# ここに他のTierを追加したい場合は、Notebookの対応する make_*_cfg 関数をコピーして
# この辞書に登録すれば --task で指定できるようになる
TASKS = {
    "baseline": baseline_env_cfg,
    "tier2a_domain_rand": domain_rand_env_cfg,
    "tier3": command_env_cfg,
}


def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str):
    from rsl_rl.models.mlp_model import MLPModel
    dummy_obs = {"actor": torch.zeros(1, obs_dim, device=device)}
    actor = MLPModel(
        obs=dummy_obs, obs_groups={"actor": ["actor"]}, obs_set="actor",
        output_dim=act_dim, hidden_dims=(64, 64), activation="elu",
        distribution_cfg={"class_name": "rsl_rl.modules.distribution.GaussianDistribution",
                           "init_std": 1.0, "std_type": "scalar"},
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor


class PolicyWrapper:
    """NativeMujocoViewer が要求する policy(obs) -> action の形にラップする。"""

    def __init__(self, actor):
        self.actor = actor

    def __call__(self, obs):
        with torch.no_grad():
            return self.actor({"actor": obs})


def main():
    parser = argparse.ArgumentParser(description="Cartpole mjlab local interactive play")
    parser.add_argument("--checkpoint", required=True, help="学習済みモデル(.pt)のパス")
    parser.add_argument("--task", choices=list(TASKS.keys()), default="tier3")
    parser.add_argument("--gpu-id", type=int, default=0, help="-1でCPU")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    print(f"[play_local] task={args.task} device={device}")

    env_cfg = TASKS[args.task](num_envs=1)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    obs_dim = env.observation_space.spaces["actor"].shape[-1]
    act_dim = env.action_space.shape[-1]
    actor = load_policy(args.checkpoint, obs_dim, act_dim, device)
    policy = PolicyWrapper(actor)

    has_command = "cart_target" in getattr(env.command_manager, "active_terms", {})
    command_term = env.command_manager.get_term("cart_target") if has_command else None
    target_value = {"v": 0.0}

    def key_callback(keycode: int):
        """GLFWキーコード: 262=Right, 263=Left。← / → で目標カート位置を操作する。"""
        if command_term is None:
            return
        if keycode == 262:
            target_value["v"] = min(target_value["v"] + 0.1, 1.2)
        elif keycode == 263:
            target_value["v"] = max(target_value["v"] - 0.1, -1.2)
        else:
            return
        command_term.set_external_target(torch.full((1,), target_value["v"], device=device))
        print(f"[key] 目標カート位置: {target_value['v']:.2f}")

    viewer = NativeMujocoViewer(
        env=env,
        policy=policy,
        key_callback=key_callback,
        enable_perturbations=True,  # Ctrl+右クリックドラッグで外力を加えられる(標準機能)
    )

    print("ビューアを起動します。ウィンドウを閉じるかQキーで終了します。")
    if hasattr(viewer, "setup"):
        viewer.setup()
    try:
        running = True
        while running:
            running = viewer.tick()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(viewer, "close"):
            viewer.close()


if __name__ == "__main__":
    main()
