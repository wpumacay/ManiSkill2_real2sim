from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import sapien.core as sapien

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv

XY_MIN = (-0.35, -0.02)
XY_MAX = (-0.12, 0.42)

PLATE_SCALE = 1.0

FRUITS_SCALES = {
    'apple': 1.0,
    'orange': 1.0,
    'eggplant': 1.0,
}

FRUITS_ASSET_NAMES = {
    'apple': 'apple',
    'orange': 'orange',
    'eggplant': 'eggplant',
}


class PlayWithFruitsInSceneEnv(CustomSceneEnv):
    obj: sapien.Actor  # target object to grasp

    def __init__(
        self,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.fruits: Dict[str, sapien.Actor] = {}
        self.plate: Optional[sapien.Actor] = None

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)


    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret["control_mode"] = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_coke_can_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["overhead_camera"]

        return ret

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.20],
            "init_rot_quat": [0, 0, 0, 1],
        }
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()

    def _load_model(self) -> None:
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        reconfigure = options.get("reconfigure", False)
        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure

        options["reconfigure"] = reconfigure

        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def _initialize_actors(self):
        pass

    def evaluate(self, **kwargs):
        return dict(
            success=False,
        )

    def get_language_instruction(self, **kwargs):
        return ""

# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #


@register_env("PlayWithFruitsCustomInScene-v0", max_episode_steps=80)
class PlayWithFruitsCustomInSceneEnv(PlayWithFruitsInSceneEnv, CustomOtherObjectsInSceneEnv):
    
    def _load_model(self) -> None:
        assert self._scene is not None
        self.plate = self._build_actor_helper(
            "bridge_plate_objaverse",
            self._scene,
            scale=PLATE_SCALE,
            root_dir=self.asset_root.as_posix(),
        )
        self.plate.set_pose(sapien.Pose(p=np.array([-0.235, 0.2, self.scene_table_height + 0.05])))


        self.fruits['apple'] = self._build_fruit_helper(
            pos=np.array([-0.125, 0.4, self.scene_table_height + 0.05]),
            fruit_id='apple'
        )
        self.fruits['orange'] = self._build_fruit_helper(
            pos=np.array([-0.125, 0.0, self.scene_table_height + 0.05]),
            fruit_id='orange'
        )
        self.fruits['eggplant'] = self._build_fruit_helper(
            pos=np.array([-0.325, 0.4, self.scene_table_height + 0.05]),
            fruit_id='eggplant'
        )

    def _build_fruit_helper(self, pos: np.ndarray, fruit_id: str) -> sapien.Actor:
        assert self._scene is not None
        fruit_scale = FRUITS_SCALES[fruit_id]
        fruit_asset = FRUITS_ASSET_NAMES[fruit_id]

        fruit = self._build_actor_helper(
            fruit_asset,
            self._scene,
            scale=fruit_scale,
            root_dir=self.asset_root.as_posix(),
        )
        fruit.set_pose(sapien.Pose(p=pos))
        return fruit
