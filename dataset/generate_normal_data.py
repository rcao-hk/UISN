import numpy as np
import os
from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
import argparse
import open3d as o3d
import multiprocessing


def generate_scene(scene_id, cfgs):
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'normals')
    for ann_id in range(256):
        print('generating scene: {} ann: {}'.format(scene_id, ann_id))
        depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                    camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
        seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'label', str(ann_id).zfill(4) + '.png')))
        meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # remove outlier and get objectness label
        depth_mask = (depth > 0)
        camera_poses = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'camera_poses.npy'))
        camera_pose = camera_poses[ann_id]
        align_mat = np.load(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'cam0_wrt_table.npy'))
        trans = np.dot(align_mat, camera_pose)
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        cloud_masked = cloud[mask]

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked.reshape(-1, 3))
        scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=True)
        scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        scene.normalize_normals()
        normal_masked = np.asarray(scene.normals).astype(np.float16)

        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, str(ann_id).zfill(4) + '.npy'), normal_masked)


def parallel_generate(scene_ids, cfgs, proc = 2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes = proc)
    for scene_id in scene_ids:
        p.apply_async(generate_scene, (scene_id, cfgs))
    p.close()
    p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet')
    parser.add_argument('--camera_type', default='realsense', help='Camera split [realsense/kinect]')
    cfgs = parser.parse_args()
    
    parallel_generate(list(range(190)), cfgs=cfgs, proc = 20)