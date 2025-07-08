from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv

CUBE_SIZE = 0.025

XY_MIN = (-0.35, -0.02)
XY_MAX = (-0.12, 0.42)


class PlayWithCubesInSceneEnv(CustomSceneEnv):
    obj: sapien.Actor  # target object to grasp

    def __init__(
        self,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.cubes: List[sapien.Actor] = []

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
        cube1 = self._build_cube_helper(
            pos=np.array([-0.235, 0.2, self.scene_table_height + 0.05]),
            color=np.array([1.0, 0.0, 0.0]),
        )
        cube2 = self._build_cube_helper(
            pos=np.array([-0.125, 0.0, self.scene_table_height + 0.05]),
            color=np.array([0.0, 1.0, 0.0]),
        )
        cube3 = self._build_cube_helper(
            pos=np.array([-0.325, 0.4, self.scene_table_height + 0.05]),
            color=np.array([0.0, 0.0, 1.0]),
        )

        self.cubes.append(cube1)
        self.cubes.append(cube2)
        self.cubes.append(cube3)

    def _build_cube_helper(self, pos: np.ndarray, color: np.ndarray) -> sapien.Actor:
        assert self._scene is not None
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(
            half_size=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
        )
        builder.add_box_visual(
            half_size=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            color=color,
        )

        cube = builder.build()
        cube.set_pose(sapien.Pose(p=pos))
        return cube

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
        return "pick blue cube"

# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #


@register_env("PlayWithCubesCustomInScene-v0", max_episode_steps=80)
class PlayWithCubesCustomInSceneEnv(PlayWithCubesInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass

