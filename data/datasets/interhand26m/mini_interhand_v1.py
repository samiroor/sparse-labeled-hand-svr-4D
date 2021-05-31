import os
import json
import glob
import pickle
import hashlib
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO

from ..videos import Videos
from ..torch_handmodel import Torch_HandModel
from .interhand_video import InterHandVideo


def get_joints_reorder_map(input_order, target_order):
  """
  input_order (dict): {joint_name:joint_id}
  target_order (dict): {joint_name:joint_id}
  """
  ids_map = list()
  target_order = dict(map(reversed, target_order.items()))
  joints_num = len(target_order)
  # right hand
  for jn_id in range(joints_num):
    finger_part = target_order[jn_id]
    ids_map.append(input_order['r_'+f'{finger_part}'])
  # left hand
  for jn_id in range(joints_num):
    finger_part = target_order[jn_id]
    ids_map.append(input_order['l_'+f'{finger_part}'])
  return np.array(ids_map, dtype=np.int8)


class MiniInterHand_v1(Videos):
  def __init__(self,
               cfg,
               split="train",
               supervise_fraction=1.0,
               add_first_last_frame=True,
               mano_root=None,
               images_features=None,
               val_idx=25,
               **kwargs,
               ):
    """
    Args:
      cfg (EasyDict): of dataset cfg. Must have the following items:
        - db_root (str): path of dataset.
        - annot_metric (str): deesired metric of annotations ("m"|"mm").
        - crop_bbox (bool): cropping frames around the hand.
        - crop_factor (float): desired tightness of cropped frames around the hand.
        - supervise_fraction (float): limit the annotations availablity.
        - add_first_last_frame (bool): to include first-last frames in supervised set frames
        - use_mini_dataset (bool): use mini_dataset or not
        - mini_dataset (EasyDict):
            *** mini_dataset would be chosen from train split.
            train (list of str): list of video_names for "train" split.
            val (list of str): list of video_names for "val" split.
            test (list of str): list of video_names for "test" split.
      split (str): desired dataset split "train"|"test"|"val".
      mano_root (str)
      images_features (dict): {"video_id:frame_id:filename": corresponding features}
    """
    super().__init__()
    self.split = split
    self.supervise_fraction = supervise_fraction
    self.add_first_last_frame = add_first_last_frame
    self.images_features = images_features

    # get data root; video_folders; object_meshes; hand_model
    self.db_root = os.path.join(cfg.db_root, "InterHand2.6M/mini-interhand30fps-v1")
    self.annot_path = os.path.join(self.db_root, 'annotations')

    # dataset specifics.
    self.min_len_video = 1000
    self._ref_handside = 'right'
    self._annot_metric = cfg.get('annot_metric', 'm')
    self._root_joint = cfg.get('root_joint', 0)
    self._ref_bone = cfg.get('ref_bone', [8,9])
    self._crop_bbox = cfg.get('crop_bbox', False)
    self._crop_factor = cfg.get('crop_factor', 1.2)

    self._image_size = [512,512] #[512, 334] #some images are [334, 512] so we pad smaller side.
    self._joints = {w: i
                    for i, w in enumerate([
                      'wrist',
                      'thumb1', 'thumb2', 'thumb3', 'thumb4',
                      'index1', 'index2', 'index3', 'index4',
                      'middle1', 'middle2', 'middle3', 'middle4',
                      'ring1', 'ring2', 'ring3', 'ring4',
                      'pinky1', 'pinky2', 'pinky3', 'pinky4'])}
    self._bones = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [0, 5],
            [5, 6],
            [6, 7],
            [7, 8],
            [0, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [0, 13],
            [13, 14],
            [14, 15],
            [15, 16],
            [0, 17],
            [17, 18],
            [18, 19],
            [19, 20]]

    self._torch_hand_model = {"right":Torch_HandModel(mano_root,
                                                      use_pca=False,
                                                      use_mean_pose=True,
                                                      ncomps=45,
                                                      ref_side="right",
                                                      center_idx=self._root_joint,
                                                      annot_metric=self._annot_metric),
                              "left":Torch_HandModel(mano_root,
                                                     use_pca=False,
                                                     use_mean_pose=True,
                                                     ncomps=45,
                                                     ref_side="left",
                                                     center_idx=self._root_joint,
                                                     annot_metric=self._annot_metric)}

    # joints order
    joints_order = dict()
    with open(os.path.join(self.annot_path, 'skeleton.txt')) as f:
      for line in f:
        if line[0] == "#":continue
        joint_name,joint_id,_ = line.split(' ')
        joints_order[joint_name] = joint_id
    reorder_idx = get_joints_reorder_map(joints_order, self._joints)

    # load annotations
    cache_path = os.path.join(
        self.annot_path,
        'test_annotations.pkl' if self.split=='test' else 'train_annotations.pkl'
        )
    with open(cache_path, 'rb') as f:
      annots = pickle.load(f)

    self.joints = annots['joints']
    self.cameras = annots['cameras']
    self.mano_params = annots['mano_params']

    if self.split == 'train':
      video_names = annots['video_names'][:-val_idx]
    elif self.split == 'val':
      video_names = annots['video_names'][-val_idx:]
    if self.split == 'test':
      video_names = annots['video_names'] # len(annots) == len(video_names)

    video_folders  = dict(filter(lambda elem:elem[0] in video_names, annots['video_folders'].items()))
    # arrange frames of each video
    for video_dir in video_folders:
      video_folders[video_dir] = sorted(video_folders[video_dir], key=lambda elem:int(elem['frame_idx']))

    # create video objects
    self.videos = list()
    for video_dir, anns in video_folders.items():
      campos,camrot = self.get_video_camposrot(anns[0])
      focal, princpt = self.get_video_focalprincpt(anns[0])
      video_joints = self.get_video_joints3d(anns[0])
      video_mano_params = self.get_video_mano_params(anns[0])
      self.videos.append(InterHandVideo(video_dir=os.path.join(self.db_root, video_dir),
                                        anns=anns,
                                        campos=campos,camrot=camrot,
                                        focal=focal,princpt=princpt,
                                        video_joints=video_joints,
                                        reorder_idx=reorder_idx,
                                        video_mano_params=video_mano_params,
                                        image_size=self._image_size,
                                        annot_metric=self._annot_metric,
                                        crop_bbox=self._crop_bbox,
                                        crop_factor=self._crop_factor,
                                        supervise_fraction=self.supervise_fraction,
                                        add_first_last_frame=self.add_first_last_frame))
    # release memory 
    self.joints = None
    self.cameras = None
    self.mano_params = None
    video_folders = None

    # list of ids ['vid_id:frm_id', ...]
    self.list_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._all_frames]
    self.list_supervised_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._supervised_frames]
    self.list_unsupervised_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._unsupervised_frames]

    self.num_videos = len(self.videos)
    self.num_videos_frames = len(self.list_videos_frames)
    self.num_supervised_videos_frames = len(self.list_supervised_videos_frames)
    self.num_unsupervised_videos_frames = len(self.list_unsupervised_videos_frames)

    print(f"Mini-InterHand2.6M-v1:{self.split}\n \
            num_videos_frames:{self.num_videos_frames}\n \
            num_supervised_videos_frames:{self.num_supervised_videos_frames}\n \
            num_unsupervised_videos_frames:{self.num_unsupervised_videos_frames}")

    # images features
    if self.images_features is not None:
      assert len(self.images_features)==len(self.list_videos_frames),"Images features are not matched with data."

  def ann2name(self, ann):
    return f"Capture{ann['capture']}/{ann['seq_name']}/cam{ann['camera']}/"

  def get_video_focalprincpt(self, ann):
    capture = ann['capture']; camera = ann['camera']
    focal = np.array(self.cameras[str(capture)]['focal'][str(camera)], dtype=np.float32)
    princpt = np.array(self.cameras[str(capture)]['princpt'][str(camera)], dtype=np.float32)
    return focal, princpt

  def get_video_camposrot(self, ann):
    capture = ann['capture']; camera = ann['camera']
    campos = np.array(self.cameras[str(capture)]['campos'][str(camera)], dtype=np.float32)
    camrot = np.array(self.cameras[str(capture)]['camrot'][str(camera)], dtype=np.float32)
    return campos,camrot

  def get_video_joints3d(self, ann):
    capture = ann['capture']; camera = ann['camera']
    return self.joints[str(capture)]

  def get_video_mano_params(self, ann):
    capture = ann['capture']; camera = ann['camera']
    return self.mano_params[str(capture)]
