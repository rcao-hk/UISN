import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2, 3"

import numpy as np
import open3d as o3d

from PIL import Image
import scipy.io as scio
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.data_utils import get_workspace_mask, CameraInfo, create_point_cloud_from_depth_image
from pytorch3d.ops.knn import knn_points

import torch
from suctionnetAPI.utils.xmlhandler import xmlReader
from suctionnetAPI.utils.eval_utils import parse_posevector, eval_suction, create_table_points, voxel_sample_points, \
    transform_points, get_scene_name

import argparse
import multiprocessing

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)
partial_num = 100000


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def loadSealLabels(dataset_root, objIds=None):
    '''
    **Input:**
    - objIds: int or list of int of the object ids.
    **Output:**
    - a dict of seal labels of each object.
    '''
    # load object-level grasp labels of the given obj ids
    assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
    objIds = objIds if _isArrayLike(objIds) else [objIds]
    graspLabels = {}
    # for i in tqdm(objIds, desc='Loading seal labels...'):
    for i in objIds:
        file = np.load(os.path.join(dataset_root, 'seal_label', '{}_seal.npz'.format(str(i).zfill(3))))
        graspLabels[i] = (file['points'].astype(np.float32), file['normals'].astype(np.float32), file['scores'].astype(np.float32))
    return graspLabels


def loadWrenchLabels(dataset_root, sceneIds):
    '''
    **Input:**

    - sceneIds: int or list of int of the scene ids.
    **Output:**
    - dict of the wrench labels.
    '''
    assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
    sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
    wrenchLabels = {}
    # for sid in tqdm(sceneIds, desc='Loading wrench labels...'):
    for sid in sceneIds:
        labels = np.load(os.path.join(dataset_root, 'wrench_label', '%04d_wrench.npz' % sid))
        wrenchLabel = []
        for j in range(len(labels)):
            wrenchLabel.append(labels['arr_{}'.format(j)])
        wrenchLabels['scene_'+str(sid).zfill(4)] = wrenchLabel
    return wrenchLabels


def loadCollisionLabels(dataset_root, sceneIds):
    '''
    **Input:**

    - sceneIds: int or list of int of the scene ids.
    **Output:**
    - dict of the collision labels.
    '''
    assert _isArrayLike(sceneIds) or isinstance(sceneIds, int), 'sceneIds must be an integer or a list/numpy array of integers'
    sceneIds = sceneIds if _isArrayLike(sceneIds) else [sceneIds]
    collisionLabels = {}
    # for sid in tqdm(sceneIds, desc='Loading collision labels...'):
    for sid in sceneIds:
        labels = np.load(os.path.join(dataset_root, 'suction_collision_label', '%04d_collision.npz' % sid))
        collisionLabel = []
        for j in range(len(labels)):
            collisionLabel.append(labels['arr_{}'.format(j)])
        collisionLabels['scene_'+str(sid).zfill(4)] = collisionLabel
    return collisionLabels


def get_model_poses(dataset_root, camera, scene_id, ann_id):
    '''
    **Input:**

    - scene_id: int of the scen index.

    - ann_id: int of the annotation index.

    **Output:**

    - obj_list: list of int of object index.

    - pose_list: list of 4x4 matrices of object poses.

    - camera_pose: 4x4 matrix of the camera pose relative to the first frame.

    - align mat: 4x4 matrix of camera relative to the table.
    '''
    scene_dir = os.path.join(dataset_root, 'scenes')
    camera_poses_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'camera_poses.npy')
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[ann_id]
    align_mat_path = os.path.join(dataset_root, 'scenes', get_scene_name(scene_id), camera, 'cam0_wrt_table.npy')
    align_mat = np.load(align_mat_path)
    scene_reader = xmlReader(os.path.join(scene_dir, get_scene_name(scene_id), camera, 'annotations', '%04d.xml'% (ann_id,)))
    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    return obj_list, pose_list, camera_pose, align_mat


def point_matching(scene_points, grasp_points, seal_scores, wrench_scores):
    grasp_points = grasp_points.contiguous().unsqueeze(0)
    scene_points_num = scene_points.shape[0]
    scene_seal_scores = np.zeros((scene_points_num, 1))
    scene_wrench_scores = np.zeros((scene_points_num, 1))
    part_num = int(scene_points_num / partial_num)
    for i in range(1, part_num + 2):   # lack of cuda memory
        if i == part_num + 1:
            cloud_masked_partial = scene_points[partial_num * part_num:]
            if len(cloud_masked_partial) == 0:
                break
        else:
            cloud_masked_partial = scene_points[partial_num * (i - 1):(i * partial_num)]
        cloud_masked_partial = torch.from_numpy(cloud_masked_partial).cuda().float()
        cloud_masked_partial = cloud_masked_partial.contiguous().unsqueeze(0)
        _, nn_inds, _ = knn_points(cloud_masked_partial, grasp_points, K=1)
        nn_inds = nn_inds.squeeze(-1).squeeze(0)
        scene_seal_scores[partial_num * (i - 1):(i * partial_num)] = torch.index_select(seal_scores, 0, nn_inds).cpu().numpy()
        scene_wrench_scores[partial_num * (i - 1):(i * partial_num)] = torch.index_select(wrench_scores, 0, nn_inds).cpu().numpy()
    return scene_seal_scores, scene_wrench_scores


def generate_scene(scene_id, cfgs):
    dataset_root = cfgs.dataset_root   # set dataset root
    camera_type = cfgs.camera_type   # kinect / realsense
    save_path_root = os.path.join(dataset_root, 'suction')
    os.makedirs(save_path_root, exist_ok=True)
    for ann_id in range(256):
        # get scene point cloud
        print('generating scene: {} ann: {}'.format(scene_id, ann_id))
        rgb = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'rgb', str(ann_id).zfill(4) + '.png')))
        depth = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                                camera_type, 'depth', str(ann_id).zfill(4) + '.png')))
        seg = np.array(Image.open(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                            camera_type, 'label', str(ann_id).zfill(4) + '.png')))

        meta = scio.loadmat(os.path.join(dataset_root, 'scenes', 'scene_' + str(scene_id).zfill(4),
                                        camera_type, 'meta', str(ann_id).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        obj_list, pose_list, camera_pose, align_mat = get_model_poses(dataset_root, camera_type, scene_id, ann_id)
        workspace_mask = get_workspace_mask(cloud, seg=seg, trans=np.dot(align_mat, camera_pose), organized=True, outlier=0.02)

        depth_mask = (depth > 0)
        mask = (depth_mask & workspace_mask)
        # print(obj_list)

        rgb_masked = rgb[mask]
        cloud_masked = cloud[mask]
        objectness_label = seg[mask]

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(cloud_masked.reshape((-1, 3)))
        scene.colors = o3d.utility.Vector3dVector(rgb_masked/255.0)
        downsampled_scene = scene.voxel_down_sample(voxel_size=0.003)

        seal_labels = loadSealLabels(dataset_root, obj_list)
        wrench_labels = loadWrenchLabels(dataset_root, scene_id)
        wrench_dump = wrench_labels['scene_{:04}'.format(scene_id)]

        grasp_points = []
        grasp_points_seal = []
        grasp_points_wrench = []
        inst_vis_list = []
        for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
            sampled_points, sampled_normals, seal_scores = seal_labels[obj_idx]
            wrench_scores = wrench_dump[i]

            target_points = transform_points(sampled_points, trans)
            grasp_points.append(target_points)
            grasp_points_seal.append(seal_scores.reshape(-1, 1))
            grasp_points_wrench.append(wrench_scores.reshape(-1, 1))
            # grasp_points_graspness.append(graspness.reshape(num_points, 1))

            inst_vis = o3d.geometry.PointCloud()
            inst_vis.points = o3d.utility.Vector3dVector(target_points.reshape((-1, 3)))
            inst_vis_list.append(inst_vis)

        grasp_points = np.vstack(grasp_points)
        grasp_points_seal = np.vstack(grasp_points_seal)
        grasp_points_wrench = np.vstack(grasp_points_wrench)

        # grasp_points_graspness = np.vstack(grasp_points_graspness)
        #
        grasp_points = torch.from_numpy(grasp_points).cuda()
        grasp_points_seal = torch.from_numpy(grasp_points_seal).cuda()
        grasp_points_wrench = torch.from_numpy(grasp_points_wrench).cuda()

        cloud_masked_seal_scores, cloud_masked_wrench_scores = point_matching(cloud_masked, grasp_points,
                                                                            grasp_points_seal, grasp_points_wrench)

        # max_seal_scores = np.max(cloud_masked_seal_scores)
        # min_seal_scores = np.min(cloud_masked_seal_scores)
        # cloud_masked_seal_scores = (cloud_masked_seal_scores - min_seal_scores) / (max_seal_scores - min_seal_scores)
        cloud_masked_seal_scores[~objectness_label.astype(bool), :] = 0

        # max_wrench_scores = np.max(cloud_masked_wrench_scores)
        # min_wrench_scores = np.min(cloud_masked_wrench_scores)
        # cloud_masked_wrench_scores = (cloud_masked_wrench_scores - min_wrench_scores) / (max_wrench_scores - min_wrench_scores)
        cloud_masked_wrench_scores[~objectness_label.astype(bool), :] = 0

        save_path = os.path.join(save_path_root, 'scene_' + str(scene_id).zfill(4), camera_type)
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(os.path.join(save_path, str(ann_id).zfill(4) + '.npz'),
                            seal_score=cloud_masked_seal_scores, wrench_score=cloud_masked_wrench_scores)
            
    
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
    
    parallel_generate(list(range(190)), cfgs=cfgs, proc = 10)