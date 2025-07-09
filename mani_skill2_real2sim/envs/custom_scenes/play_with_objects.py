from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import sapien.core as sapien

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.registration import register_env

from .base_env import CustomOtherObjectsInSceneEnv

from scipy.spatial.transform import Rotation as R

XY_MIN = (-0.35, -0.02)
XY_MAX = (-0.12, 0.42)


class PlayWithObjectsInSceneEnv(CustomOtherObjectsInSceneEnv):
    objects_info: Dict[str, Any]

    def __init__(
        self,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        self.objects: Dict[str, sapien.Actor] = {}

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
        self._load_models()

    def _load_models(self) -> None:
        assert self._scene is not None
        for obj_id in self.objects_info:
            self.objects[obj_id] = self._build_object_helper(object_id=obj_id)

    def _build_object_helper(self, object_id: str) -> sapien.Actor:
        assert self._scene is not None
        object_scale = self.objects_info[object_id]['scale']
        object_asset = self.objects_info[object_id]['asset_id']
        object_init_xy = self.objects_info[object_id]['init_xy']
        object_init_rot = self.objects_info[object_id].get('init_rot', [0.0, 0.0, 0.0])
        object_height_offset = self.objects_info[object_id].get('height_offset', 0.05)
        z = self.scene_table_height + object_height_offset

        obj = self._build_actor_helper(
            object_asset,
            self._scene,
            scale=object_scale,
            root_dir=self.asset_root.as_posix(),
        )
        obj.set_pose(
            sapien.Pose(
                p=np.array([object_init_xy[0], object_init_xy[1], z]),
                q=R.from_euler('xyz', object_init_rot, degrees=True).as_quat(scalar_first=False)
            )
        )
        return obj

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

@register_env("PlayWithObjectsCustomInScene-v0", max_episode_steps=80)
class PlayWithObjectsCustomInSceneV0Env(PlayWithObjectsInSceneEnv):
    objects_info = {
        'plate': {
            'scale': 1.0,
            'asset_id': 'bridge_plate_objaverse',
            'init_xy': [-0.235, 0.2],
        },
        'spoon': {
            'scale': 1.0,
            'asset_id': 'bridge_spoon_generated_modified',
            'init_xy': [-0.125, 0.0],
        },
        'apple': {
            'scale': 1.0,
            'asset_id': 'apple',
            'init_xy': [-0.325, 0.4],
        }
    }

@register_env("PlayWithObjectsCustomInScene-v1", max_episode_steps=80)
class PlayWithObjectsCustomInSceneV1Env(PlayWithObjectsInSceneEnv):
    objects_info = {
        'plate': {
            'scale': 1.5,
            'asset_id': 'bridge_plate_objaverse',
            'init_xy': [-0.235, 0.2],
        },
        'bowl': {
            'scale': 1.0,
            'asset_id': 'eli_bowl',
            'init_xy': [-0.125, 0.0],
            'init_rot': [0.0, 0.0, 90.0],
            'height_offset': 0.05,
        },
        'redbull': {
            'scale': 1.0,
            'asset_id': 'redbull_can',
            'init_xy': [-0.325, 0.4],
            'init_rot': [0.0, 180.0, 90.0],
            'height_offset': 0.15,
        }
    }


@register_env("PlayWithObjectsCustomInScene-v2", max_episode_steps=80)
class PlayWithObjectsCustomInSceneV2Env(PlayWithObjectsInSceneEnv):
    objects_info = {
        'mug': {
            'scale': 1.0,
            'asset_id': 'eli_mug',
            'init_xy': [-0.235, 0.2],
            'init_rot': [0.0, 0.0, 90.0],
        },
        'marker': {
            'scale': 1.0,
            'asset_id': 'eli_marker',
            'init_xy': [-0.325, 0.4],
            'height_offset': 0.15,
        }
    }


