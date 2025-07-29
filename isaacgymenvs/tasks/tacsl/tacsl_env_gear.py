
"""TacSL: class for gear insertion env.

Inherits TacSL base class and abstract environment class. Inherited by insertion task class. Not directly executed.

Configuration defined in TacSLEnvGear.yaml. Asset info defined in industreal_asset_info_pegs.yaml.

need change industreal_asset_info_pegs.yaml
"""

from collections import defaultdict
import hydra
import numpy as np
import os
import torch
import cv2

from isaacgym import gymapi
from isaacgym.torch_utils import tf_combine
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tacsl_sensors.tacsl_sensors import CameraSensor, TactileRGBSensor, TactileFieldSensor
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from isaacgymenvs.tasks.tacsl.tacsl_base import TacSLBase
from isaacgymenvs.tacsl_sensors.shear_tactile_viz_utils import visualize_penetration_depth, visualize_tactile_shear_image


class TacSLSensors(TactileFieldSensor, TactileRGBSensor, CameraSensor):

    def get_regular_camera_specs(self):
        # TODO: Maybe this should go inside the task files, as it depends on task configs e.g. self.cfg_task.env.use_camera

        camera_spec_dict = {}
        if self.cfg_task.env.use_camera:
            camera_spec_dict = {c_cfg.name: c_cfg for c_cfg in self.cfg_task.env.camera_configs}
        return camera_spec_dict

    def _compose_tactile_image_configs(self):
        tactile_sensor_config = {
            'tactile_camera_name': 'left_tactile_camera',
            'actor_name': 'franka',
            'actor_handle': self.actor_handles['franka'],
            'attach_link_name': 'elastomer_tip_left',
            'elastomer_link_name': 'elastomer_left',
            'compliance_stiffness': self.cfg_task.env.compliance_stiffness,
            'compliant_damping': self.cfg_task.env.compliant_damping,
            'use_acceleration_spring': False,
            'sensor_type': 'gelsight_r15'
        }
        tactile_sensor_config_left = tactile_sensor_config.copy()
        tactile_sensor_config_right = tactile_sensor_config.copy()
        tactile_sensor_config_right['tactile_camera_name'] = 'right_tactile_camera'
        tactile_sensor_config_right['attach_link_name'] = 'elastomer_tip_right'
        tactile_sensor_config_right['elastomer_link_name'] = 'elastomer_right'
        tactile_sensor_configs = [tactile_sensor_config_left, tactile_sensor_config_right]
        return tactile_sensor_configs

    def _compose_tactile_force_field_configs(self):
        plug_rb_names = self.gym.get_actor_rigid_body_names(self.env_ptrs[0], self.plug_actor_id_env)
        tactile_shear_field_config = dict([
            ('name', 'tactile_force_field_left'),
            ('elastomer_actor_name', 'franka'), ('elastomer_link_name', 'elastomer_left'),
            ('elastomer_tip_link_name', 'elastomer_tip_left'),
            ('elastomer_parent_urdf_path', self.asset_file_paths["franka"]),
            ('indenter_urdf_path', self.asset_file_paths["plug"]),
            ('indenter_actor_name', 'plug'), ('indenter_link_name', plug_rb_names[0]),
            ('actor_handle', self.actor_handles['franka']),
            ('compliance_stiffness', self.cfg_task.env.compliance_stiffness),
            ('compliant_damping', self.cfg_task.env.compliant_damping),
            ('use_acceleration_spring', False)
        ])
        tactile_shear_field_config_left = tactile_shear_field_config.copy()
        tactile_shear_field_config_right = tactile_shear_field_config.copy()
        tactile_shear_field_config_right['name'] = 'tactile_force_field_right'
        tactile_shear_field_config_right['elastomer_link_name'] = 'elastomer_right'
        tactile_shear_field_config_right['elastomer_tip_link_name'] = 'elastomer_tip_right'
        tactile_shear_field_configs = [tactile_shear_field_config_left, tactile_shear_field_config_right]
        return tactile_shear_field_configs

    def get_tactile_force_field_tensors_dict(self):

        tactile_force_field_dict_raw = self.get_tactile_shear_force_fields()
        tactile_force_field_dict_processed = dict()
        tactile_depth_dict_processed = dict()
        nrows, ncols = self.cfg_task.env.num_shear_rows, self.cfg_task.env.num_shear_cols

        debug = False   # Debug visualization
        for k in tactile_force_field_dict_raw:
            penetration_depth, tactile_normal_force, tactile_shear_force = tactile_force_field_dict_raw[k]
            tactile_force_field = torch.cat(
                (tactile_normal_force.view((self.num_envs, nrows, ncols, 1)),
                 tactile_shear_force.view((self.num_envs, nrows, ncols, 2))),
                dim=-1)
            tactile_force_field_dict_processed[k] = tactile_force_field
            tactile_depth_dict_processed[k] = penetration_depth.view((self.num_envs, nrows, ncols))

            if debug:
                env_viz_id = 0
                tactile_image = visualize_tactile_shear_image(
                    tactile_normal_force[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    tactile_shear_force[env_viz_id].view((nrows, ncols, 2)).cpu().numpy(),
                    normal_force_threshold=0.0008,
                    shear_force_threshold=0.0008)
                cv2.imshow(f'Force Field {k}', tactile_image.swapaxes(0, 1))

                penetration_depth_viz = visualize_penetration_depth(
                    penetration_depth[env_viz_id].view((nrows, ncols)).cpu().numpy(),
                    resolution=5, depth_multiplier=300.)
                cv2.imshow(f'FF Penetration Depth {k}', penetration_depth_viz.swapaxes(0, 1))
        return tactile_force_field_dict_processed, tactile_depth_dict_processed

    def _create_sensors(self):
        self.camera_spec_dict = dict()
        self.camera_handles_list = []
        self.camera_tensors_list = []
        if self.cfg_task.env.use_isaac_gym_tactile:
            tactile_sensor_configs = self._compose_tactile_image_configs()
            self.set_compliant_dynamics_for_tactile_sensors(tactile_sensor_configs)
            camera_spec_dict_tactile = self.get_tactile_rgb_camera_configs(tactile_sensor_configs)
            self.camera_spec_dict.update(camera_spec_dict_tactile)

        if self.cfg_task.env.use_camera:
            camera_spec_dict = self.get_regular_camera_specs()
            self.camera_spec_dict.update(camera_spec_dict)

        if self.camera_spec_dict:
            # tactile cameras created along with other cameras in create_camera_actors
            camera_handles_list, camera_tensors_list = self.create_camera_actors(self.camera_spec_dict)
            self.camera_handles_list += camera_handles_list
            self.camera_tensors_list += camera_tensors_list

        if self.cfg_task.env.get('use_shear_force', False):
            tactile_ff_configs = self._compose_tactile_force_field_configs()
            self.set_compliant_dynamics_for_tactile_sensors(tactile_ff_configs)
            self.sdf_tool = 'physx'
            self.sdf_tensor = self.setup_tactile_force_field(self.sdf_tool,
                                                             self.cfg_task.env.num_shear_rows,
                                                             self.cfg_task.env.num_shear_cols,
                                                             tactile_ff_configs)


class TacSLEnvGear(TacSLBase, TacSLSensors, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()
        self.nominal_tactile = None

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name="factory_schema_config_env", node=FactorySchemaConfigEnv)

        config_path = "task/TacSLEnvGear.yaml"  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env["task"]  # strip superfluous nesting

        asset_info_path = "../../assets/industreal/yaml/industreal_asset_info_gears.yaml"  # relative to Hydra search path (cfg dir)
        self.asset_info_gears = hydra.compose(config_name=asset_info_path)
        self.asset_info_gears = self.asset_info_gears[""][""][""][""][""][""]["assets"][
            "industreal"
        ][
            "yaml"
        ]  # strip superfluous nesting

    def create_envs(self):
        import inspect
        caller = inspect.stack()[1]  # Get the caller's stack frame
        print(f"Called by function: {caller.function}")

        """Set env options. Import assets. Create actors."""
        #lower - upper means bounds for env creation

        lower = gymapi.Vec3(-self.asset_info_franka_table.table_depth * 0.6,
                            -self.asset_info_franka_table.table_width * 0.6,
                            0.0)
        upper = gymapi.Vec3(self.asset_info_franka_table.table_depth * 0.6,
                            self.asset_info_franka_table.table_width * 0.6,
                            self.asset_info_franka_table.table_height)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        self.assets = dict()
        self.asset_file_paths = dict()
        self.assets['franka'], self.assets['table'] = self.import_franka_assets()
        self._import_env_assets()
        self._create_actors(lower, upper, num_per_row)
        self.parse_controller_spec()

        self._create_sensors()

    def _import_env_assets(self):
        """Set gear and base asset options. Import assets."""
        
        urdf_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "industreal", "urdf"
        )
        gear_small_file = "industreal_gear_small.urdf"
        gear_medium_file = "industreal_gear_medium.urdf"
        gear_large_file = "industreal_gear_large.urdf"
        base_file = "industreal_gear_base.urdf"

        print(f'test👌suceessfully loaded gear files')

        gear_options = gymapi.AssetOptions()
        gear_options.flip_visual_attachments = False
        gear_options.fix_base_link = False
        gear_options.thickness = 0.0  # default = 0.02
        gear_options.density = self.asset_info_gears.gears.density  # default = 1000.0
        gear_options.armature = 0.0  # default = 0.0
        gear_options.use_physx_armature = True
        gear_options.linear_damping = 0.5  # default = 0.0
        gear_options.max_linear_velocity = 1000.0  # default = 1000.0
        gear_options.angular_damping = 0.5  # default = 0.5
        gear_options.max_angular_velocity = 64.0  # default = 64.0
        gear_options.disable_gravity = True
        gear_options.enable_gyroscopic_forces = True
        gear_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        gear_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            gear_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        base_options = gymapi.AssetOptions()
        base_options.flip_visual_attachments = False
        base_options.fix_base_link = True
        base_options.thickness = 0.0  # default = 0.02
        base_options.density = self.asset_info_gears.base.density  # default = 1000.0
        base_options.armature = 0.0  # default = 0.0
        base_options.use_physx_armature = True
        base_options.linear_damping = 0.0  # default = 0.0
        base_options.max_linear_velocity = 1000.0  # default = 1000.0
        base_options.angular_damping = 0.0  # default = 0.5
        base_options.max_angular_velocity = 64.0  # default = 64.0
        base_options.disable_gravity = False
        base_options.enable_gyroscopic_forces = True
        base_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        base_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            base_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        gear_small_asset = self.gym.load_asset(
            self.sim, urdf_root, gear_small_file, gear_options
        )
        gear_medium_asset = self.gym.load_asset(
            self.sim, urdf_root, gear_medium_file, gear_options
        )
        gear_large_asset = self.gym.load_asset(
            self.sim, urdf_root, gear_large_file, gear_options
        )
        base_asset = self.gym.load_asset(self.sim, urdf_root, base_file, base_options)

        self.gear_files = [os.path.join(urdf_root, gear_medium_file)]
        self.shaft_files = [os.path.join(urdf_root, base_file)]
        # NOTE: Saving asset indices is not necessary for IndustRealEnvGears
        self.asset_indices = [0 for _ in range(self.num_envs)]
        
        self.assets['gear_small'] = gear_small_asset
        self.assets['gear_medium'] = gear_medium_asset
        self.assets['gear_large'] = gear_large_asset
        self.assets['gear_base'] = base_asset

    def _create_actors(self, lower, upper, num_per_row):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        # from self.assets get assets
        franka_asset      = self.assets['franka']
        gear_small_asset  = self.assets["gear_small"]
        gear_medium_asset = self.assets["gear_medium"]
        gear_large_asset  = self.assets["gear_large"]
        base_asset        = self.assets["gear_base"]
        table_asset       = self.assets["table"]
        
        franka_pose = gymapi.Transform()
        franka_pose.p.x = 0.0
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = self.asset_info_franka_table.robot_base_to_table_offset_x
        table_pose.p.y = 0.0
        table_pose.p.z = -self.asset_info_franka_table.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # TO-DO: Add initial gear poses paremeters to TacSLTaskGear.yaml
        # mimicking the socket pose from TacSLTaskInsertion.yaml
        gear_pose = gymapi.Transform()
        gear_pose.p.x = 0.5
        gear_pose.p.y = self.cfg_env.env.gears_lateral_offset
        gear_pose.p.z = 0.01
        gear_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # TO-DO: Add initial base poses paremeters to TacSLTaskGear.yaml
        # mimicking the socket pose from TacSLTaskInsertion.yaml
        # socket_pos_xyz_initial: [0.5, 0.0, 0.01] 
        base_pose = gymapi.Transform()
        base_pose.p.x = 0.5
        base_pose.p.y = 0.0
        base_pose.p.z = 0.01
        base_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 统一存储 actor handles 与 sim 内索引
        self.env_ptrs = []
        
        # from industreal_env_gears.py
        self.franka_handles = []
        self.gear_small_handles = []
        self.gear_medium_handles = []
        self.gear_large_handles = []
        self.base_handles = []
        self.table_handles = []
        self.shape_ids = []

        # from industreal_env_gears.py
        self.franka_actor_ids_sim = []  # within-sim indices
        self.gear_small_actor_ids_sim = []  # within-sim indices
        self.gear_medium_actor_ids_sim = []  # within-sim indices
        self.gear_large_actor_ids_sim = []  # within-sim indices
        self.base_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices

        self.env_subassembly_id = []
        self.actor_handles = {}               # 存放各个资产的 handle，键为 'franka', 'gear_small', 'gear_medium', 'gear_large', 'base', 'table'
        self.actor_ids_sim = defaultdict(list)  # 存放各个资产在 sim 中的索引
        self.rbs_com       = defaultdict(list)
        actor_count      = 0

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # creat franka robot
            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, "franka", i + self.num_envs, 0, 0
                )
            else:
                franka_handle = self.gym.create_actor(
                    env_ptr, franka_asset, franka_pose, "franka", i, 0, 0
                )
            self.actor_handles['franka'] = franka_handle
            self.actor_ids_sim['franka'].append(actor_count)
            actor_count += 1

            # creat gear_small
            gear_small_handle = self.gym.create_actor(
                env_ptr, gear_small_asset, gear_pose, "gear_small", i, 0, 0
            )
            # self.actor_handles["gear_small"] = gear_small_handle
            # self.actor_ids_sim["gear_small"].append(actor_count)
            self.gear_small_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat gear_medium
            gear_medium_handle = self.gym.create_actor(
                env_ptr, gear_medium_asset, gear_pose, "gear_medium", i, 0, 0
            )
            # self.actor_handles['gear_medium'] = gear_medium_handle
            # self.actor_ids_sim["gear_medium"].append(actor_count)
            self.gear_medium_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat gear_large
            gear_large_handle = self.gym.create_actor(
                env_ptr, gear_large_asset, gear_pose, "gear_large", i, 0, 0
            )
            # self.actor_handles['gear_large'] = gear_large_handle
            # self.actor_ids_sim["gear_large"].append(actor_count)
            self.gear_large_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat base asix object
            base_handle = self.gym.create_actor(
                env_ptr, base_asset, base_pose, "base", i, 0, 0
            )
            # self.actor_handles['base'] = base_handle
            # self.actor_ids_sim["base"].append(actor_count)
            self.base_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat table
            table_handle = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0, 0
            )
            self.actor_handles['table'] = table_handle
            # self.actor_ids_sim['table'].append(actor_count)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            # setting franka rigid body properties
            link7_id    = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_link7", gymapi.DOMAIN_ACTOR
            )
            hand_id     = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_ACTOR
            )
            left_finger_id  = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_leftfinger", gymapi.DOMAIN_ACTOR
            )
            right_finger_id = self.gym.find_actor_rigid_body_index(
                env_ptr, franka_handle, "panda_rightfinger", gymapi.DOMAIN_ACTOR
            )
            rb_ids = [link7_id, hand_id, left_finger_id, right_finger_id]
            rb_shape_indices = self.gym.get_asset_rigid_body_shape_indices(franka_asset)
            self.shape_ids = [rb_shape_indices[rb_id].start for rb_id in rb_ids]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0
                franka_shape_props[shape_id].restitution = 0.0
                franka_shape_props[shape_id].compliance = 0.0
                franka_shape_props[shape_id].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            # set gear_small rigid body properties
            gear_small_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_small_handle)
            gear_small_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_small_shape_props[0].rolling_friction = 0.0
            gear_small_shape_props[0].torsion_friction = 0.0
            gear_small_shape_props[0].restitution = 0.0
            gear_small_shape_props[0].compliance = 0.0
            gear_small_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_small_handle, gear_small_shape_props)

            # set gear_medium rigid body properties
            gear_medium_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_medium_handle)
            gear_medium_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_medium_shape_props[0].rolling_friction = 0.0
            gear_medium_shape_props[0].torsion_friction = 0.0
            gear_medium_shape_props[0].restitution = 0.0
            gear_medium_shape_props[0].compliance = 0.0
            gear_medium_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_medium_handle, gear_medium_shape_props)

            # set gear_large rigid body properties
            gear_large_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_large_handle)
            gear_large_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_large_shape_props[0].rolling_friction = 0.0
            gear_large_shape_props[0].torsion_friction = 0.0
            gear_large_shape_props[0].restitution = 0.0
            gear_large_shape_props[0].compliance = 0.0
            gear_large_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_large_handle, gear_large_shape_props)

            # set base rigid body properties
            base_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, base_handle)
            base_shape_props[0].friction = self.cfg_env.env.base_friction
            base_shape_props[0].rolling_friction = 0.0
            base_shape_props[0].torsion_friction = 0.0
            base_shape_props[0].restitution = 0.0
            base_shape_props[0].compliance = 0.0
            base_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, base_handle, base_shape_props)

            # set table rigid body properties
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0
            table_shape_props[0].torsion_friction = 0.0
            table_shape_props[0].restitution = 0.0
            table_shape_props[0].compliance = 0.0
            table_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)
            # start franka DOF force sensor
            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)
            
            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.gear_small_handles.append(gear_small_handle)
            self.gear_medium_handles.append(gear_medium_handle)
            self.gear_large_handles.append(gear_large_handle)
            self.base_handles.append(base_handle)
            self.table_handles.append(table_handle)        

        if self.cfg_task.env.use_compliant_contact:
            # Set compliance params
            self.set_elastomer_compliance(self.cfg_task.env.compliance_stiffness, self.cfg_task.env.compliant_damping)

        # update info
        self.num_actors = int(actor_count / self.num_envs)
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)
        self.num_dofs   = self.gym.get_env_dof_count(env_ptr)

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(
            self.actor_ids_sim['franka'], dtype=torch.int32, device=self.device
        )
        self.gear_small_actor_ids_sim = torch.tensor(
            self.gear_small_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.gear_medium_actor_ids_sim = torch.tensor(
            self.gear_medium_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.gear_large_actor_ids_sim = torch.tensor(
            self.gear_large_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.base_actor_ids_sim = torch.tensor(
            self.base_actor_ids_sim, dtype=torch.int32, device=self.device
        )

        self.actor_ids_sim_tensors = dict()
        self.actor_ids_sim_tensors['franka'] = self.franka_actor_ids_sim

        # # 将各资产的 sim 索引转换为 tensor（用于设置目标等）
        # self.actor_ids_sim_tensors = {
        #     key: torch.tensor(val, dtype=torch.int32, device=self.device)
        #     for key, val in self.actor_ids_sim.items()
        # }

        # For extracting root pos/quat
        self.franka_actor_id_env      = self.gym.find_actor_index(env_ptr, "franka", gymapi.DOMAIN_ENV)
        self.gear_small_actor_id_env  = self.gym.find_actor_index(env_ptr, "gear_small", gymapi.DOMAIN_ENV)
        self.gear_medium_actor_id_env = self.gym.find_actor_index(env_ptr, "gear_medium", gymapi.DOMAIN_ENV)
        self.gear_large_actor_id_env  = self.gym.find_actor_index(env_ptr, "gear_large", gymapi.DOMAIN_ENV)
        self.base_actor_id_env        = self.gym.find_actor_index(env_ptr, "base", gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_ENV
        )
        self.gear_small_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, gear_small_handle, "gear_small", gymapi.DOMAIN_ENV
        )
        self.gear_medium_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, gear_medium_handle, "gear_small", gymapi.DOMAIN_ENV
        )
        self.gear_large_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, gear_large_handle, "gear_small", gymapi.DOMAIN_ENV
        )
        self.base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, base_handle, "base", gymapi.DOMAIN_ENV
        )

        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.left_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                               'panda_leftfingertip',
                                                                               gymapi.DOMAIN_ENV)
        self.right_fingertip_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                'panda_rightfingertip',
                                                                                gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)
        self.hand_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                           gymapi.DOMAIN_ACTOR)
        self.left_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                  'panda_leftfinger',
                                                                                  gymapi.DOMAIN_ACTOR)
        self.right_finger_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_rightfinger',
                                                                                   gymapi.DOMAIN_ACTOR)
        self.left_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                     'panda_leftfingertip',
                                                                                     gymapi.DOMAIN_ACTOR)
        self.right_fingertip_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                      'panda_rightfingertip',
                                                                                      gymapi.DOMAIN_ACTOR)
        self.fingertip_centered_body_id_env_actor = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                         'panda_fingertip_centered',
                                                                                         gymapi.DOMAIN_ACTOR)
 
        self.table_body_id = self.gym.find_actor_rigid_body_index(self.env_ptrs[0], self.actor_handles['table'],
                                                                  'box', gymapi.DOMAIN_ENV)
        self.franka_body_names = self.gym.get_actor_rigid_body_names(env_ptr, franka_handle)
        self.franka_body_ids_env = dict()
        for b_name in self.franka_body_names:
            self.franka_body_ids_env[b_name] = self.gym.find_actor_rigid_body_index(self.env_ptrs[0],
                                                                                    self.actor_handles['franka'],
                                                                                    b_name, gymapi.DOMAIN_ENV)

    def set_elastomer_compliance(self, compliance_stiffness, compliant_damping):
        for elastomer_link_name in ['elastomer_left', 'elastomer_right']:
            self.configure_compliant_dynamics(actor_handle=self.actor_handles['franka'],
                                              elastomer_link_name=elastomer_link_name,
                                              compliance_stiffness=compliance_stiffness,
                                              compliant_damping=compliant_damping,
                                              use_acceleration_spring=False)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.gear_small_pos = self.root_pos[:, self.gear_small_actor_id_env, 0:3]
        self.gear_small_quat = self.root_quat[:, self.gear_small_actor_id_env, 0:4]
        self.gear_small_linvel = self.root_linvel[:, self.gear_small_actor_id_env, 0:3]
        self.gear_small_angvel = self.root_angvel[:, self.gear_small_actor_id_env, 0:3]

        self.gear_medium_pos = self.root_pos[:, self.gear_medium_actor_id_env, 0:3]
        self.gear_medium_quat = self.root_quat[:, self.gear_medium_actor_id_env, 0:4]
        self.gear_medium_linvel = self.root_linvel[
            :, self.gear_medium_actor_id_env, 0:3
        ]
        self.gear_medium_angvel = self.root_angvel[
            :, self.gear_medium_actor_id_env, 0:3
        ]

        self.gear_large_pos = self.root_pos[:, self.gear_large_actor_id_env, 0:3]
        self.gear_large_quat = self.root_quat[:, self.gear_large_actor_id_env, 0:4]
        self.gear_large_linvel = self.root_linvel[:, self.gear_large_actor_id_env, 0:3]
        self.gear_large_angvel = self.root_angvel[:, self.gear_large_actor_id_env, 0:3]

        self.base_pos = self.root_pos[:, self.base_actor_id_env, 0:3]
        self.base_quat = self.root_quat[:, self.base_actor_id_env, 0:4]
        self.base_linvel = self.root_linvel[:, self.base_actor_id_env, 0:3]
        self.base_angvel = self.root_angvel[:, self.base_actor_id_env, 0:3]

        self.gear_small_com_pos = fc.translate_along_local_z(
            pos=self.gear_small_pos,
            quat=self.gear_small_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_small_com_quat = self.gear_small_quat  # always equal
        self.gear_small_com_linvel = self.gear_small_linvel + torch.cross(
            self.gear_small_angvel,
            (self.gear_small_com_pos - self.gear_small_pos),
            dim=1,
        )
        self.gear_small_com_angvel = self.gear_small_angvel  # always equal

        self.gear_medium_com_pos = fc.translate_along_local_z(
            pos=self.gear_medium_pos,
            quat=self.gear_medium_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_medium_com_quat = self.gear_medium_quat  # always equal
        self.gear_medium_com_linvel = self.gear_medium_linvel + torch.cross(
            self.gear_medium_angvel,
            (self.gear_medium_com_pos - self.gear_medium_pos),
            dim=1,
        )
        self.gear_medium_com_angvel = self.gear_medium_angvel  # always equal

        self.gear_large_com_pos = fc.translate_along_local_z(
            pos=self.gear_large_pos,
            quat=self.gear_large_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_large_com_quat = self.gear_large_quat  # always equal
        self.gear_large_com_linvel = self.gear_large_linvel + torch.cross(
            self.gear_large_angvel,
            (self.gear_large_com_pos - self.gear_large_pos),
            dim=1,
        )
        self.gear_large_com_angvel = self.gear_large_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gear_small_com_pos = fc.translate_along_local_z(
            pos=self.gear_small_pos,
            quat=self.gear_small_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_small_com_linvel = self.gear_small_linvel + torch.cross(
            self.gear_small_angvel,
            (self.gear_small_com_pos - self.gear_small_pos),
            dim=1,
        )

        self.gear_medium_com_pos = fc.translate_along_local_z(
            pos=self.gear_medium_pos,
            quat=self.gear_medium_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_medium_com_linvel = self.gear_medium_linvel + torch.cross(
            self.gear_medium_angvel,
            (self.gear_medium_com_pos - self.gear_medium_pos),
            dim=1,
        )

        self.gear_large_com_pos = fc.translate_along_local_z(
            pos=self.gear_large_pos,
            quat=self.gear_large_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.gear_large_com_linvel = self.gear_large_linvel + torch.cross(
            self.gear_large_angvel,
            (self.gear_large_com_pos - self.gear_large_pos),
            dim=1,
        )