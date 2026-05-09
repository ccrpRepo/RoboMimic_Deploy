from common.path_config import PROJECT_ROOT

from policy.passive.PassiveMode import PassiveMode
from policy.fixedpose.FixedPose import FixedPose
from policy.loco_mode.LocoMode import LocoMode
from policy.kungfu.KungFu import KungFu
from policy.dance.Dance import Dance
from policy.skill_cooldown.SkillCooldown import SkillCooldown
from policy.skill_cast.SkillCast import SkillCast
from policy.kick.Kick import Kick
from policy.kungfu2.KungFu2 import KungFu2
from policy.beyond_mimic.BeyondMimic import BeyondMimic
from policy.holomotion.HoloMotion import HoloMotion
from FSM.FSMState import *
import time
from common.ctrlcomp import *
from enum import Enum, unique

@unique
class FSMMode(Enum):
    CHANGE = 1
    NORMAL = 2

class FSM:
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.cur_policy : FSMState
        self.next_policy : FSMState
        
        self.FSMmode = FSMMode.NORMAL
        
        self.passive_mode = PassiveMode(state_cmd, policy_output)
        self.fixed_pose_1 = FixedPose(state_cmd, policy_output)
        self.loco_policy = LocoMode(state_cmd, policy_output)
        self.kungfu_policy = KungFu(state_cmd, policy_output)
        self.dance_policy = Dance(state_cmd, policy_output)
        self.skill_cooldown_policy = SkillCooldown(state_cmd, policy_output)
        self.skill_cast_policy = SkillCast(state_cmd, policy_output)
        self.kick_policy = Kick(state_cmd, policy_output)
        self.kungfu2_policy = KungFu2(state_cmd, policy_output)
        self.beyond_mimic_policy = BeyondMimic(state_cmd, policy_output)
        self.holomotion_policy = HoloMotion(state_cmd, policy_output)
        
        print("initalized all policies!!!")
        
        self.cur_policy = self.passive_mode
        print("current policy is ", self.cur_policy.name_str)
        
        
        
    def run(self):
        start_time = time.time()
        if(self.FSMmode == FSMMode.NORMAL): 
            self.cur_policy.run()
            nextPolicyName = self.cur_policy.checkChange()
            
            if(nextPolicyName != self.cur_policy.name):
                # change policy
                self.FSMmode = FSMMode.CHANGE
                self.cur_policy.exit()
                self.get_next_policy(nextPolicyName)
                print("Switched to ", self.cur_policy.name_str)
        
        elif(self.FSMmode == FSMMode.CHANGE):
            self.cur_policy.enter()
            self.FSMmode = FSMMode.NORMAL
            self.cur_policy.run()
            
        # self.absoluteWait(self.cur_policy.control_horzion,self.start_time)
        end_time = time.time()
        # print("time cusume: ", end_time - start_time)

    def absoluteWait(self, control_dt, start_time):
        end_time = time.time()
        delta_time = end_time - start_time
        if(delta_time < control_dt):
            time.sleep(control_dt - delta_time)
        else:
            print("inference time beyond control horzion!!!")
            
            
    def get_next_policy(self, policy_name:FSMStateName):
        if(policy_name == FSMStateName.PASSIVE):
            self.cur_policy = self.passive_mode
        elif((policy_name == FSMStateName.FIXEDPOSE)):
            self.cur_policy = self.fixed_pose_1
        elif((policy_name == FSMStateName.LOCOMODE)):
            self.cur_policy = self.loco_policy
        elif((policy_name == FSMStateName.SKILL_KungFu)):
            self.cur_policy = self.kungfu_policy
        elif((policy_name == FSMStateName.SKILL_Dance)):
            self.cur_policy = self.dance_policy
        elif((policy_name == FSMStateName.SKILL_COOLDOWN)):
            self.cur_policy = self.skill_cooldown_policy
        elif((policy_name == FSMStateName.SKILL_CAST)):
            self.cur_policy = self.skill_cast_policy
        elif((policy_name == FSMStateName.SKILL_KICK)):
            self.cur_policy = self.kick_policy
        elif((policy_name == FSMStateName.SKILL_KungFu2)):
            self.cur_policy = self.kungfu2_policy
        elif((policy_name == FSMStateName.SKILL_BEYOND_MIMIC)):
            self.cur_policy = self.beyond_mimic_policy
        elif((policy_name == FSMStateName.SKILL_HOLOMOTION)):
            self.cur_policy = self.holomotion_policy
        else:
            pass
            
        
        