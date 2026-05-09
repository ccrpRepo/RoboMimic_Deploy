from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import torch
import os

class LocoMode(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.LOCOMODE
        self.name_str = "Loco_mode"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "LocoMode.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = os.path.join(current_dir, "model", config["policy_path"])
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles =  np.array(config["default_angles"], dtype=np.float32)
            self.joint2motor_idx =  np.array(config["joint2motor_idx"], dtype=np.int32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.cmd_range = config["cmd_range"]
            self.range_velx = np.array([self.cmd_range["lin_vel_x"][0], self.cmd_range["lin_vel_x"][1]], dtype=np.float32)
            self.range_vely = np.array([self.cmd_range["lin_vel_y"][0], self.cmd_range["lin_vel_y"][1]], dtype=np.float32)
            self.range_velz = np.array([self.cmd_range["ang_vel_z"][0], self.cmd_range["ang_vel_z"][1]], dtype=np.float32)
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            
            # load policy
            self.policy = torch.jit.load(self.policy_path)
            
            for _ in range(50):
                with torch.inference_mode():
                    obs_tensor = self.obs.reshape(1, -1)
                    obs_tensor = obs_tensor.astype(np.float32)
                    self.policy(torch.from_numpy(obs_tensor))
                    
            print("Locomotion policy initializing ...")
                
    
    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            self.kps_reorder[motor_idx] = self.kps[i]
            self.kds_reorder[motor_idx] = self.kds[i]
            self.default_angles_reorder[motor_idx] = self.default_angles[i]
            
    
    def run(self):
        self.gravity_orientation = self.state_cmd.gravity_ori
        self.qj = self.state_cmd.q.copy()
        self.dqj = self.state_cmd.dq.copy()
        self.ang_vel = self.state_cmd.ang_vel.copy()
        joycmd = self.state_cmd.vel_cmd.copy()
        self.cmd = scale_values(joycmd, [self.range_velx, self.range_vely, self.range_velz])
        
        for i in range(len(self.joint2motor_idx)):
            self.qj_obs[i] = self.qj[self.joint2motor_idx[i]]
            self.dqj_obs[i] = self.dqj[self.joint2motor_idx[i]]
            
        self.qj_obs = (self.qj_obs - self.default_angles) * self.dof_pos_scale
        self.dqj_obs = self.dqj_obs * self.dof_vel_scale
        self.ang_vel = self.ang_vel * self.ang_vel_scale
        self.cmd = self.cmd * self.cmd_scale
        
        self.obs[:3] = self.ang_vel.copy()
        self.obs[3:6] = self.gravity_orientation.copy()
        self.obs[6:9] = self.cmd.copy()
        self.obs[9: 9 + self.num_actions] = self.qj_obs.copy()
        self.obs[9 + self.num_actions: 9 + self.num_actions * 2] = self.dqj_obs.copy()
        self.obs[9 + self.num_actions * 2: 9 + self.num_actions * 3] = self.action.copy()
        
        obs_tensor = self.obs.reshape(1, -1)
        obs_tensor = obs_tensor.astype(np.float32)
        self.action = self.policy(torch.from_numpy(obs_tensor).clip(-100, 100)).clip(-100, 100).detach().numpy().squeeze()
        loco_action = self.action * self.action_scale + self.default_angles
        action_reorder = loco_action.copy()
        for i in range(len(self.joint2motor_idx)):
            motor_idx = self.joint2motor_idx[i]
            action_reorder[motor_idx] = loco_action[i]
            
        
        self.policy_output.actions = action_reorder.copy()
        self.policy_output.kps = self.kps_reorder.copy()
        self.policy_output.kds = self.kds_reorder.copy()
        # print("actions: ", self.policy_output.actions)
    
    def exit(self):
        pass
    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.SKILL_1):
            return FSMStateName.SKILL_Dance
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_2):
            return FSMStateName.SKILL_KungFu
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_3):
            return FSMStateName.SKILL_KICK
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_4):
            return FSMStateName.SKILL_BEYOND_MIMIC
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_5):
            return FSMStateName.SKILL_HOLOMOTION
        else:
            return FSMStateName.LOCOMODE
        