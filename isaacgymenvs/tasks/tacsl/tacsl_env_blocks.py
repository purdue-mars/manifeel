
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
from isaacgymenvs.tasks.tacsl.tacsl_base_gear import TacSLBaseGear
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


class TacSLEnvBlocks(TacSLBaseGear, TacSLSensors, FactoryABCEnv):

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

        config_path = "task/TacSLEnvBlocks.yaml"  # relative to Gym's Hydra search path (cfg dir)
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

        block_part1_file   = "blocks_part1.urdf" #gear_small_asset
        block_part2_file   = "blocks_part2.urdf" #new
        block_sec_last_file= "blocks_part3.urdf" #gear_large_asset
        block_last_file    = "blocks_part4.urdf" #gear_medium_asset        
        base_file          = "blocks_base.urdf"

        # print(f'test👌suceessfully loaded 3d model files')

        blocks_options = gymapi.AssetOptions()
        blocks_options.flip_visual_attachments = False
        blocks_options.fix_base_link = False
        blocks_options.thickness     = 0.0  # default = 0.02
        blocks_options.density = self.asset_info_gears.gears.density  # default = 1000.0
        blocks_options.armature      = 0.0  # default = 0.0
        blocks_options.use_physx_armature   = True
        blocks_options.linear_damping       = 0.5  # default = 0.0
        blocks_options.max_linear_velocity  = 1000.0  # default = 1000.0
        blocks_options.angular_damping      = 0.5  # default = 0.5
        blocks_options.max_angular_velocity = 64.0  # default = 64.0
        blocks_options.disable_gravity      = False
        blocks_options.enable_gyroscopic_forces = True
        blocks_options.default_dof_drive_mode   = gymapi.DOF_MODE_NONE
        blocks_options.use_mesh_materials   = False
        if self.cfg_base.mode.export_scene:
            blocks_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        base_options = gymapi.AssetOptions()
        base_options.flip_visual_attachments  = False
        base_options.fix_base_link        = True
        base_options.thickness            = 0.0  # default = 0.02
        base_options.density = self.asset_info_gears.base.density  # default = 1000.0
        base_options.armature             = 0.0  # default = 0.0
        base_options.use_physx_armature   = True
        base_options.linear_damping       = 0.0  # default = 0.0
        base_options.max_linear_velocity  = 1000.0  # default = 1000.0
        base_options.angular_damping      = 0.0  # default = 0.5
        base_options.max_angular_velocity = 64.0  # default = 64.0
        base_options.disable_gravity      = False
        base_options.enable_gyroscopic_forces = True
        base_options.default_dof_drive_mode   = gymapi.DOF_MODE_NONE
        base_options.use_mesh_materials   = False
        if self.cfg_base.mode.export_scene:
            base_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        block_part1_asset = self.gym.load_asset(
            self.sim, urdf_root, block_part1_file, blocks_options
        )
        block_part2_asset = self.gym.load_asset(
            self.sim, urdf_root, block_part2_file, blocks_options
        )        
        block_last_asset = self.gym.load_asset(
            self.sim, urdf_root, block_last_file, blocks_options
        )
        block_sec_last_asset = self.gym.load_asset(
            self.sim, urdf_root, block_sec_last_file, blocks_options
        )
        base_asset = self.gym.load_asset(self.sim, urdf_root, base_file, base_options)

        self.gear_files = [os.path.join(urdf_root, block_last_file)]
        self.shaft_files = [os.path.join(urdf_root, base_file)]
        # NOTE: Saving asset indices is not necessary for IndustRealEnvGears
        self.asset_indices = [0 for _ in range(self.num_envs)]

        self.assets['block_part1']    = block_part1_asset
        self.assets['block_part2']    = block_part2_asset        
        self.assets['block_last']     = block_last_asset
        self.assets['block_sec_last'] = block_sec_last_asset
        self.assets['block_base']     = base_asset

    def _create_actors(self, lower, upper, num_per_row):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        # from self.assets get assets
        franka_asset         = self.assets['franka']
        block_part1_asset    = self.assets["block_part1"]
        block_part2_asset    = self.assets["block_part2"]
        block_sec_last_asset = self.assets["block_sec_last"]
        block_last_asset     = self.assets["block_last"]        
        base_asset           = self.assets["block_base"]
        table_asset          = self.assets["table"]        
        
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
        blocks_pose = gymapi.Transform()
        blocks_pose.p.x = 0.5
        blocks_pose.p.y = self.cfg_env.env.gears_lateral_offset
        blocks_pose.p.z = 0.01
        blocks_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # TO-DO: Add initial base poses paremeters to TacSLTaskGear.yaml
        # mimicking the socket pose from TacSLTaskInsertion.yaml
        # socket_pos_xyz_initial: [0.5, 0.0, 0.01] 
        base_pose = gymapi.Transform()
        base_pose.p.x = 0.5
        base_pose.p.y = 0.0
        base_pose.p.z = 0.01
        base_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.env_ptrs = []
        
        # from industreal_env_gears.py
        self.franka_handles = []
        self.base_handles   = []
        self.table_handles  = []
        self.shape_ids      = []
        self.block_part1    = []
        self.block_part2    = []
        self.block_sec_last = []
        self.block_last     = []

        # from industreal_env_gears.py
        self.franka_actor_ids_sim         = []  # within-sim indices
        self.block_part1_actor_ids_sim    = []  
        self.block_part2_actor_ids_sim    = []
        self.block_sec_last_actor_ids_sim = []
        self.block_last_actor_ids_sim     = []
        self.base_actor_ids_sim           = []  
        self.table_actor_ids_sim          = []  

        self.env_subassembly_id = []
        self.actor_handles = {}               
        self.actor_ids_sim = defaultdict(list)  
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

            # creat block_part1
            block_part1_handle = self.gym.create_actor(
                env_ptr, block_part1_asset, blocks_pose, "block_part1", i, 0, 0
            ) 
            self.block_part1_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat block_part2
            block_part2_handle = self.gym.create_actor(
                env_ptr, block_part2_asset, blocks_pose, "block_part2", i, 0, 0
            ) 
            self.block_part2_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat block_sec_last
            block_sec_last_handle = self.gym.create_actor(
                env_ptr, block_sec_last_asset, blocks_pose, "block_sec_last", i, 0, 0
            ) 
            self.block_sec_last_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat block_last
            block_last_handle = self.gym.create_actor(
                env_ptr, block_last_asset, blocks_pose, "block_last", i, 0, 0
            ) 
            self.block_last_actor_ids_sim.append(actor_count)
            actor_count += 1

            # creat base asix object
            base_handle = self.gym.create_actor(
                env_ptr, base_asset, base_pose, "base", i, 0, 0
            )
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

            # set block_part1 rigid body properties
            block_part1_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_part1_handle)
            block_part1_shape_props[0].friction = self.cfg_env.env.gears_friction
            block_part1_shape_props[0].rolling_friction = 0.0
            block_part1_shape_props[0].torsion_friction = 0.0
            block_part1_shape_props[0].restitution = 0.0
            block_part1_shape_props[0].compliance = 0.0
            block_part1_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_part1_handle, block_part1_shape_props)

            # set block_part2 rigid body properties
            block_part2_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_part2_handle)
            block_part2_shape_props[0].friction = self.cfg_env.env.gears_friction
            block_part2_shape_props[0].rolling_friction = 0.0
            block_part2_shape_props[0].torsion_friction = 0.0
            block_part2_shape_props[0].restitution = 0.0
            block_part2_shape_props[0].compliance = 0.0
            block_part2_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_part2_handle, block_part2_shape_props)

            # set block_sec_last rigid body properties
            block_sec_last_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_sec_last_handle)
            block_sec_last_shape_props[0].friction = self.cfg_env.env.gears_friction
            block_sec_last_shape_props[0].rolling_friction = 0.0
            block_sec_last_shape_props[0].torsion_friction = 0.0
            block_sec_last_shape_props[0].restitution = 0.0
            block_sec_last_shape_props[0].compliance = 0.0
            block_sec_last_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_sec_last_handle, block_sec_last_shape_props)

            # set block_last rigid body properties
            block_last_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_last_handle)
            block_last_shape_props[0].friction = self.cfg_env.env.gears_friction
            block_last_shape_props[0].rolling_friction = 0.0
            block_last_shape_props[0].torsion_friction = 0.0
            block_last_shape_props[0].restitution = 0.0
            block_last_shape_props[0].compliance = 0.0
            block_last_shape_props[0].thickness = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_last_handle, block_last_shape_props)

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
            self.table_handles.append(table_handle)
            self.block_part1.append(block_part1_handle)
            self.block_part2.append(block_part2_handle)
            self.block_sec_last.append(block_sec_last_handle)
            self.block_last.append(block_last_handle)
            self.base_handles.append(base_handle)        

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
        self.block_part1_actor_ids_sim = torch.tensor(
            self.block_part1_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.block_part2_actor_ids_sim = torch.tensor(
            self.block_part2_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.block_last_actor_ids_sim = torch.tensor(
            self.block_last_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.block_sec_last_actor_ids_sim = torch.tensor(
            self.block_sec_last_actor_ids_sim, dtype=torch.int32, device=self.device
        )
        self.base_actor_ids_sim = torch.tensor(
            self.base_actor_ids_sim, dtype=torch.int32, device=self.device
        )

        self.actor_ids_sim_tensors = dict()
        self.actor_ids_sim_tensors['franka'] = self.franka_actor_ids_sim

        # For extracting root pos/quat
        self.franka_actor_id_env      = self.gym.find_actor_index(env_ptr, "franka", gymapi.DOMAIN_ENV)
        self.block_part1_actor_id_env  = self.gym.find_actor_index(env_ptr, "block_part1", gymapi.DOMAIN_ENV)
        self.block_part2_actor_id_env  = self.gym.find_actor_index(env_ptr, "block_part2", gymapi.DOMAIN_ENV)
        self.block_sec_last_actor_id_env  = self.gym.find_actor_index(env_ptr, "block_sec_last", gymapi.DOMAIN_ENV)
        self.block_last_actor_id_env  = self.gym.find_actor_index(env_ptr, "block_last", gymapi.DOMAIN_ENV)                
        # self.gear_medium_actor_id_env = self.gym.find_actor_index(env_ptr, "gear_medium", gymapi.DOMAIN_ENV)
        # self.gear_large_actor_id_env  = self.gym.find_actor_index(env_ptr, "gear_large", gymapi.DOMAIN_ENV)
        self.base_actor_id_env        = self.gym.find_actor_index(env_ptr, "base", gymapi.DOMAIN_ENV)
        # self.blocks_part2_actor_id_env      = self.gym.find_actor_index(env_ptr, "blocks_part2", gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.robot_base_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_ENV
        )
        self.block_part1_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, block_part1_handle, "block_part1", gymapi.DOMAIN_ENV
        )
        self.block_part2_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, block_part2_handle, "block_part2", gymapi.DOMAIN_ENV
        )
        self.block_last_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, block_last_handle, "block_last", gymapi.DOMAIN_ENV
        )
        self.block_sec_last_body_id_env = self.gym.find_actor_rigid_body_index(
            env_ptr, block_sec_last_handle, "block_sec_last", gymapi.DOMAIN_ENV
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
        self.block_part1_pos    = self.root_pos[:, self.block_part1_actor_id_env, 0:3]
        self.block_part1_quat   = self.root_quat[:, self.block_part1_actor_id_env, 0:4]
        self.block_part1_linvel = self.root_linvel[:, self.block_part1_actor_id_env, 0:3]
        self.block_part1_angvel = self.root_angvel[:, self.block_part1_actor_id_env, 0:3]

        self.block_part2_pos    = self.root_pos[:, self.block_part2_actor_id_env, 0:3]
        self.block_part2_quat   = self.root_quat[:, self.block_part2_actor_id_env, 0:4]
        self.block_part2_linvel = self.root_linvel[:, self.block_part2_actor_id_env, 0:3]
        self.block_part2_angvel = self.root_angvel[:, self.block_part2_actor_id_env, 0:3]

        self.block_sec_last_pos    = self.root_pos[:, self.block_sec_last_actor_id_env, 0:3]
        self.block_sec_last_quat   = self.root_quat[:, self.block_sec_last_actor_id_env, 0:4]
        self.block_sec_last_linvel = self.root_linvel[:, self.block_sec_last_actor_id_env, 0:3]
        self.block_sec_last_angvel = self.root_angvel[:, self.block_sec_last_actor_id_env, 0:3]

        self.block_last_pos    = self.root_pos[:, self.block_last_actor_id_env, 0:3]
        self.block_last_quat   = self.root_quat[:, self.block_last_actor_id_env, 0:4]
        self.block_last_linvel = self.root_linvel[:, self.block_last_actor_id_env, 0:3]
        self.block_last_angvel = self.root_angvel[:, self.block_last_actor_id_env, 0:3]        

        self.base_pos = self.root_pos[:, self.base_actor_id_env, 0:3]
        self.base_quat = self.root_quat[:, self.base_actor_id_env, 0:4]
        self.base_linvel = self.root_linvel[:, self.base_actor_id_env, 0:3]
        self.base_angvel = self.root_angvel[:, self.base_actor_id_env, 0:3]

        self.block_part1_com_pos = fc.translate_along_local_z(
            pos=self.block_part1_pos,
            quat=self.block_part1_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_part1_com_quat = self.block_part1_quat  # always equal
        self.block_part1_com_linvel = self.block_part1_linvel + torch.cross(
            self.block_part1_angvel,
            (self.block_part1_com_pos - self.block_part1_pos),
            dim=1,
        )
        self.block_part1_com_angvel = self.block_part1_angvel  # always equal

        self.block_part2_com_pos = fc.translate_along_local_z(
            pos=self.block_part2_pos,
            quat=self.block_part2_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_part2_com_quat = self.block_part2_quat  # always equal
        self.block_part2_com_linvel = self.block_part2_linvel + torch.cross(
            self.block_part2_angvel,
            (self.block_part2_com_pos - self.block_part2_pos),
            dim=1,
        )
        self.block_part2_com_angvel = self.block_part2_angvel  # always equal

        self.block_sec_last_com_pos = fc.translate_along_local_z(
            pos=self.block_sec_last_pos,
            quat=self.block_sec_last_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_sec_last_com_quat = self.block_sec_last_quat  # always equal
        self.block_sec_last_com_linvel = self.block_sec_last_linvel + torch.cross(
            self.block_sec_last_angvel,
            (self.block_sec_last_com_pos - self.block_sec_last_pos),
            dim=1,
        )
        self.block_sec_last_com_angvel = self.block_sec_last_angvel  # always equal        

        self.block_last_com_pos = fc.translate_along_local_z(
            pos=self.block_last_pos,
            quat=self.block_last_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_last_com_quat = self.block_last_quat  # always equal
        self.block_last_com_linvel = self.block_last_linvel + torch.cross(
            self.block_last_angvel,
            (self.block_last_com_pos - self.block_last_pos),
            dim=1,
        )
        self.block_last_com_angvel = self.block_last_angvel  # always equal 

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        # block_part1 center position and linear velocity
        self.block_part1_com_pos = fc.translate_along_local_z(
            pos=self.block_part1_pos,
            quat=self.block_part1_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_part1_com_linvel = self.block_part1_linvel + torch.cross(
            self.block_part1_angvel,
            (self.block_part1_com_pos - self.block_part1_pos),
            dim=1,
        )

        # block_part2
        self.block_part2_com_pos = fc.translate_along_local_z(
            pos=self.block_part2_pos,
            quat=self.block_part2_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_part2_com_linvel = self.block_part2_linvel + torch.cross(
            self.block_part2_angvel,
            (self.block_part2_com_pos - self.block_part2_pos),
            dim=1,
        )

        # block_sec_last 
        self.block_sec_last_com_pos = fc.translate_along_local_z(
            pos=self.block_sec_last_pos,
            quat=self.block_sec_last_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_sec_last_com_linvel = self.block_sec_last_linvel + torch.cross(
            self.block_sec_last_angvel,
            (self.block_sec_last_com_pos - self.block_sec_last_pos),
            dim=1,
        )

        # block_last 
        self.block_last_com_pos = fc.translate_along_local_z(
            pos=self.block_last_pos,
            quat=self.block_last_quat,
            offset=self.asset_info_gears.base.height
            + self.asset_info_gears.gears.height * 0.5,
            device=self.device,
        )
        self.block_last_com_linvel = self.block_last_linvel + torch.cross(
            self.block_last_angvel,
            (self.block_last_com_pos - self.block_last_pos),
            dim=1,
        )
