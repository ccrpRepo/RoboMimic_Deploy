import json
import os
from collections import deque

import h5py
import numpy as np
import onnxruntime
import yaml

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import FSMCommand
from FSM.FSMState import FSMState, FSMStateName


def scale_velocity_command(values, target_ranges):
    """Scale velocity command values to target ranges."""
    return np.array(
        [
            min(max(val, new_min), new_max)
            for val, (new_min, new_max) in zip(values, target_ranges)
        ],
        dtype=np.float32,
    )


class HoloMotion(FSMState):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_HOLOMOTION
        self.name_str = "holomotion"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "HoloMotion.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Load basic configuration parameters
        self._load_basic_config(config)

        # Initialize state variables
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffers = {}

        # Motion tracking variables
        self.motion_frame_idx = 0
        self.current_motion_idx = 0
        self.n_motion_frames = 0
        self.ref_dof_pos = None
        self.ref_dof_vel = None
        self.motion_clips = []

        # Validate and set command mode
        self.command_mode = config.get("command_mode", "motion_tracking")
        if self.command_mode not in ["motion_tracking", "velocity_tracking"]:
            raise ValueError(
                f"Invalid command_mode: {self.command_mode}. "
                f"Must be 'motion_tracking' or 'velocity_tracking'"
            )

        # Load ONNX model and validate input dimensions
        input_dim = self._load_onnx_model(current_dir, config)

        # Initialize mode-specific parameters
        if self.command_mode == "motion_tracking":
            self._init_motion_tracking_mode(config)
        else:  # velocity_tracking
            self._init_velocity_tracking_mode(config)

        # Validate input dimensions
        if input_dim is not None and input_dim != self.num_obs:
            raise ValueError(
                f"Input dimension mismatch: ONNX model has {input_dim} dims, "
                f"but {self.command_mode} mode expects {self.num_obs} dims"
            )

        # Initialize observation buffers
        self._init_obs_buffers()

        print(
            f"HoloMotion policy initializing ... Mode: {self.command_mode}, Input dim: {input_dim}"
        )

    def _load_basic_config(self, config):
        """Load basic configuration parameters."""
        self.hdf5_root = config["hdf5_root"]
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.joint2motor_idx = np.array(config["joint2motor_idx"], dtype=np.int32)
        self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        self.dof_names_onnx = config["dof_names_onnx"]
        self.default_angles_onnx = np.array(
            config["default_angles_onnx"], dtype=np.float32
        )

    def _load_onnx_model(self, current_dir, config):
        """Load ONNX model and return input dimension."""
        onnx_key = (
            "onnx_path_motion_tracking"
            if self.command_mode == "motion_tracking"
            else "onnx_path_velocity_tracking"
        )
        self.onnx_path = os.path.join(current_dir, "model", config[onnx_key])
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)

        model_inputs = self.ort_session.get_inputs()
        self.input_name = [inpt.name for inpt in model_inputs]
        return model_inputs[0].shape[1]

    def _init_motion_tracking_mode(self, config):
        """Initialize motion tracking mode parameters."""
        self.context_length = config["context_length_motion"]
        self.num_obs = config["num_obs_motion"]
        self._load_motion_data()

    def _init_velocity_tracking_mode(self, config):
        """Initialize velocity tracking mode parameters."""
        self.context_length = config["context_length_velocity"]
        self.num_obs = config["num_obs_velocity"]
        self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        cmd_range = config["cmd_range"]
        self.range_velx = np.array(cmd_range["lin_vel_x"], dtype=np.float32)
        self.range_vely = np.array(cmd_range["lin_vel_y"], dtype=np.float32)
        self.range_velz = np.array(cmd_range["ang_vel_z"], dtype=np.float32)
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)

    def _init_obs_buffers(self):
        """Initialize observation buffers based on command mode."""
        common_buffers = {
            "projected_gravity": (deque(maxlen=self.context_length), 3),
            "rel_robot_root_ang_vel": (deque(maxlen=self.context_length), 3),
            "dof_pos": (deque(maxlen=self.context_length), self.num_actions),
            "dof_vel": (deque(maxlen=self.context_length), self.num_actions),
            "last_action": (deque(maxlen=self.context_length), self.num_actions),
        }

        if self.command_mode == "motion_tracking":
            self.obs_buffers = {
                "ref_motion_states": (
                    deque(maxlen=self.context_length),
                    2 * self.num_actions,
                ),
                **common_buffers,
            }
        else:
            self.obs_buffers = {
                "velocity_command": (deque(maxlen=self.context_length), 4),
                **common_buffers,
            }

        self.obs = np.zeros(self.num_obs, dtype=np.float32)

    def _load_motion_data(self):
        """Load motion data from HDF5 dataset."""
        manifest_path = os.path.join(self.hdf5_root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        self.dof_names_ref_motion = manifest["dof_names"]

        # Build mapping from ONNX DOF names to reference motion DOF indices
        self._build_dof_mapping(manifest)

        # Load motion clips
        self.motion_clips = list(manifest.get("clips", {}).keys())
        if len(self.motion_clips) == 0:
            raise ValueError("No motion clips found in manifest")

        # Load first motion clip as example
        self._load_motion_clip(self.motion_clips[0])

    def _build_dof_mapping(self, manifest):
        """Build mapping from ONNX DOF names to reference motion DOF indices."""
        name_mapping = {
            "torso_yaw_joint": "waist_yaw_joint",
            "torso_roll_joint": "waist_roll_joint",
            "torso_pitch_joint": "waist_pitch_joint",
        }

        self.ref_to_onnx = []
        missing_names = []

        for name in self.dof_names_onnx:
            # Try direct match first
            if name in self.dof_names_ref_motion:
                self.ref_to_onnx.append(self.dof_names_ref_motion.index(name))
            # Try mapped name (torso -> waist)
            elif (
                name in name_mapping and name_mapping[name] in self.dof_names_ref_motion
            ):
                self.ref_to_onnx.append(
                    self.dof_names_ref_motion.index(name_mapping[name])
                )
            else:
                missing_names.append(name)

        if missing_names:
            raise ValueError(
                f"DOF names not found in reference motion: {missing_names}\n"
                f"ONNX names ({len(self.dof_names_onnx)}): {self.dof_names_onnx}\n"
                f"HDF5 names ({len(self.dof_names_ref_motion)}): {self.dof_names_ref_motion}\n"
                f"Available HDF5 DOF names: {self.dof_names_ref_motion}"
            )

    def _load_motion_clip(self, motion_key: str):
        """Load a specific motion clip from HDF5."""
        manifest_path = os.path.join(self.hdf5_root, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        clip_info = manifest["clips"][motion_key]
        shard_file = os.path.join(
            self.hdf5_root, manifest["hdf5_shards"][clip_info["shard"]]["file"]
        )

        with h5py.File(shard_file, "r") as f:
            start, length = clip_info["start"], clip_info["length"]
            self.ref_dof_pos = f["dof_pos"][start : start + length].astype(np.float32)
            self.ref_dof_vel = f["dof_vels"][start : start + length].astype(np.float32)

        self.n_motion_frames = length
        self.motion_frame_idx = 0

    def _build_observation(
        self,
        q,
        dq,
        gravity_ori,
        ang_vel,
        ref_dof_pos=None,
        ref_dof_vel=None,
        velocity_command=None,
        move_mask=None,
    ):
        """Build observation from current state."""
        q_onnx, dq_onnx = self._map_joint_states_to_onnx(q, dq)
        buffer_data = self._build_mode_specific_obs(
            gravity_ori.astype(np.float32),
            ang_vel.astype(np.float32),
            q_onnx.astype(np.float32),
            dq_onnx.astype(np.float32),
            self.last_action.astype(np.float32),
            ref_dof_pos,
            ref_dof_vel,
            velocity_command,
            move_mask,
        )
        self._update_obs_buffers(buffer_data)
        return self._concatenate_obs_history()

    def _map_joint_states_to_onnx(self, q, dq):
        """Map joint states from motor order to ONNX order."""
        q_onnx = np.zeros(self.num_actions, dtype=np.float32)
        dq_onnx = np.zeros(self.num_actions, dtype=np.float32)
        for i, motor_idx in enumerate(self.joint2motor_idx):
            if i < len(q) and motor_idx < len(q):
                q_onnx[i] = q[motor_idx] - self.default_angles_onnx[i]
                dq_onnx[i] = dq[motor_idx]
        return q_onnx, dq_onnx

    def _build_mode_specific_obs(
        self,
        projected_gravity,
        rel_robot_root_ang_vel,
        dof_pos,
        dof_vel,
        last_action,
        ref_dof_pos,
        ref_dof_vel,
        velocity_command,
        move_mask,
    ):
        """Build mode-specific observation components."""
        common_data = {
            "projected_gravity": projected_gravity,
            "rel_robot_root_ang_vel": rel_robot_root_ang_vel,
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "last_action": last_action,
        }

        if self.command_mode == "motion_tracking":
            if ref_dof_pos is None or ref_dof_vel is None:
                raise ValueError(
                    "ref_dof_pos and ref_dof_vel must be provided for motion_tracking mode"
                )

            ref_dof_pos_onnx = ref_dof_pos[self.joint2motor_idx].astype(np.float32)
            ref_dof_vel_onnx = ref_dof_vel[self.joint2motor_idx].astype(np.float32)
            ref_motion_states = np.concatenate(
                [ref_dof_pos_onnx, ref_dof_vel_onnx]
            ).astype(np.float32)
            return {"ref_motion_states": ref_motion_states, **common_data}
        else:
            extended_velo_command = np.zeros(4, dtype=np.float32)
            extended_velo_command[0] = (
                float(move_mask) if move_mask is not None else 0.0
            )
            extended_velo_command[1:] = velocity_command
            return {"velocity_command": extended_velo_command, **common_data}

    def _update_obs_buffers(self, buffer_data):
        """Update observation buffers with new data."""
        for key in self.obs_buffers.keys():
            data = buffer_data[key]
            queue, _ = self.obs_buffers[key]
            queue.append(data.copy())

    def _concatenate_obs_history(self):
        """Concatenate historical observations into a single vector."""
        obs_list = [
            np.array(list(queue), dtype=np.float32).reshape(-1)
            for queue, _ in self.obs_buffers.values()
        ]
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def enter(self):
        """Initialize state when entering."""
        self._reset_obs_buffers()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_frame_idx = 0

        if self.command_mode == "motion_tracking" and len(self.motion_clips) > 0:
            self.current_motion_idx = (self.current_motion_idx + 1) % len(
                self.motion_clips
            )
            print(
                f"Loading motion clip: {self.motion_clips[self.current_motion_idx]} ({self.current_motion_idx}/{len(self.motion_clips)})"
            )
            self._load_motion_clip(self.motion_clips[self.current_motion_idx])

    def _reset_obs_buffers(self):
        """Reset all observation buffers."""
        for key, (queue, dim) in self.obs_buffers.items():
            queue.clear()
            for _ in range(self.context_length):
                queue.append(np.zeros(dim, dtype=np.float32))

    def run(self):
        """Execute policy inference."""
        state = self.state_cmd
        if self.command_mode == "motion_tracking":
            hist_obs = self._build_motion_tracking_obs(
                state.q.copy(),
                state.dq.copy(),
                state.gravity_ori.copy(),
                state.ang_vel.copy(),
            )
        else:
            hist_obs = self._build_velocity_tracking_obs(
                state.q.copy(),
                state.dq.copy(),
                state.gravity_ori.copy(),
                state.ang_vel.copy(),
            )

        observation = {self.input_name[0]: hist_obs.reshape(1, -1).astype(np.float32)}
        self.action = self.ort_session.run(None, observation)[0].squeeze()
        self._apply_action_to_output()
        self.last_action = self.action.copy()

        if self.command_mode == "motion_tracking":
            self.motion_frame_idx = (self.motion_frame_idx + 1) % self.n_motion_frames

    def _build_motion_tracking_obs(self, q, dq, gravity_ori, ang_vel):
        """Build observation for motion tracking mode."""
        frame_idx = min(self.motion_frame_idx, self.n_motion_frames - 1)
        return self._build_observation(
            q,
            dq,
            gravity_ori,
            ang_vel,
            ref_dof_pos=self.ref_dof_pos[frame_idx],
            ref_dof_vel=self.ref_dof_vel[frame_idx],
        )

    def _build_velocity_tracking_obs(self, q, dq, gravity_ori, ang_vel):
        """Build observation for velocity tracking mode."""
        joycmd = self.state_cmd.vel_cmd.copy()
        move_mask = np.linalg.norm(joycmd) > 0.1

        self.cmd = (
            scale_velocity_command(
                joycmd, [self.range_velx, self.range_vely, self.range_velz]
            )
            * self.cmd_scale
        )

        return self._build_observation(
            q,
            dq,
            gravity_ori,
            ang_vel,
            velocity_command=self.cmd.copy(),
            move_mask=move_mask,
        )

    def _apply_action_to_output(self):
        """Apply action to policy output."""
        target_dof_pos = self.action * self.action_scale + self.default_angles
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_mj[self.joint2motor_idx] = target_dof_pos

        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.joint2motor_idx] = self.kps
        self.policy_output.kds[self.joint2motor_idx] = self.kds

    def exit(self):
        """Clean up when exiting."""
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.motion_frame_idx = 0
        self._reset_obs_buffers()

    def checkChange(self):
        """Check if state should change."""
        if self.state_cmd.skill_cmd == FSMCommand.LOCO:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.LOCOMODE
        elif self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_HOLOMOTION
