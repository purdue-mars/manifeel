"""
Usage:
Visualize first 3 episodes
EPISODES="0,1,2,3,4,5,6,7,8,9" python data_vis_real_data.py --config-name data_vis_peginhole.yaml
"""

import os
import sys
import rerun as rr
from tqdm import tqdm
import numpy as np
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
import cv2
import json
import hashlib
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.replay_buffer import ReplayBuffer


def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    image_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb' and key != 'tactile_0':
            image_keys.append(key)
            c, h, w = shape
            out_resolutions[key] = (w, h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = shape
    
    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=image_keys
        )

    return replay_buffer

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

EPISODES = os.getenv("EPISODES")

AXES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

dataset_path = 'data/class_ball_pokuang_real_data_Apr12_group2'
use_cache = True
shape_meta = {
    "obs": {
        "camera_0": {
            "shape": [3, 240, 320],
            "type": "rgb"
        },
        "tactile_0": {
            "shape": [3, 240, 320],
            "type": "rgb"
        },
        "robot_eef_pose": {
            "shape": [16],
            "type": "low_dim"
        }
    },
    "action": {
        "shape": [5]
    }
}


replay_buffer = None
if use_cache:
    # fingerprint shape_meta
    shape_meta_json = json.dumps(shape_meta, sort_keys=True)
    shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
    cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
    cache_lock_path = cache_zarr_path + '.lock'
    print('Acquiring lock on cache.')
    with FileLock(cache_lock_path):
        if not os.path.exists(cache_zarr_path):
            # cache does not exists
            try:
                print('Cache does not exist. Creating!')
                replay_buffer = _get_replay_buffer(
                    dataset_path=dataset_path,
                    shape_meta=shape_meta,
                    store=zarr.MemoryStore()
                )
                print('Saving cache to disk.')
                with zarr.ZipStore(cache_zarr_path) as zip_store:
                    replay_buffer.save_to_store(
                        store=zip_store
                    )
            except Exception as e:
                shutil.rmtree(cache_zarr_path)
                raise e
        else:
            print('Loading cached ReplayBuffer from Disk.')
            with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store, store=zarr.MemoryStore())
            print('Loaded!')
else:
    replay_buffer = _get_replay_buffer(
        dataset_path=dataset_path,
        shape_meta=shape_meta,
        store=zarr.MemoryStore()
    )

print("action shape before", replay_buffer['action'].shape)
action_idxs_to_keep = [0, 1, 2, 5, 6]  # x, y, z, yaw, gripper
# Safely update the action dataset
if 'action' in replay_buffer and hasattr(replay_buffer['action'], 'resize'):
    zarr_resize_index_last_dim(replay_buffer['action'], action_idxs_to_keep)
else:
    print("Warning: 'action' dataset is not resizable. Consider using a new dataset or recreating ReplayBuffer.")
print("action shape after", replay_buffer['action'].shape)

rr.init("vistac_real_exp", spawn=False)
save_path = "debug_vistac_real_exp.rrd"
# if save_path.exists():
#     save_path.unlink()
rr.save(str(save_path))

episode_idxs = [int(EPISODE.strip()) for EPISODE in EPISODES.split(",")]

for episode_idx in tqdm(episode_idxs, "Loading episodes"):
    vis_episode = replay_buffer.get_episode(episode_idx)
    action_buffer = vis_episode["action"]
    # qpos_buffer = vis_episode["qpos"]
    size = action_buffer.shape[0]

    action_dim = action_buffer.shape[1]
    for i in range(size):
        for j in range(action_dim):
                name = AXES[j]
                rr.log(f"action/{name}", rr.Scalar(action_buffer[i, j]))
                # rr.log(f"qpos/{name}", rr.Scalar(qpos_buffer[i, j]))

        for name in [
            "camera_0",
            "tactile_0",
        ]:
            if name not in vis_episode:
                continue
            img = vis_episode[name][i]
            print(f"💾 img.shape: {img.shape}")
            rr.log(f"image/{name}", rr.Image(img))
        
        rr.log("episode", rr.Scalar(episode_idx))

rr.disconnect()
print(f"Saved at {save_path}!")