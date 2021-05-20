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


class InterHand(Videos):
  def __init__(self,
               cfg,
               split="train",
               mano_root=None,
               images_features=None,
               use_cache=True,
               cache_folder='cache/interhand2.6m_30fps'):
    """
    Args:
      cfg (EasyDict): of dataset cfg. Must have the following items:
        - db_root (str): path of dataset.
        - annot_subset (str): 'human_annot'|'machine_annot'|'all'
        - ref_handside (str): 'left'|'right'|'interacting'|'all'
        - min_len_video (int): minimum number of frames in video
        - annot_metric (str): deesired metric of annotations ("m"|"mm").
        - crop_bbox (bool): cropping frames around the hand.
        - crop_factor (float): desired tightness of cropped frames around the hand.
        - supervise_fractions (float): limit the annotations availablity.
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
    super(InterHand, self).__init__()
    self.cfg = cfg
    self.split = split

    # get data root; video_folders; object_meshes; hand_model
    self.db_root = os.path.join(cfg.db_root, "InterHand2.6M/InterHand2.6M_30fps_batch1")
    self.annot_subset = cfg.get('annot_subset','all') # human_annot, machine_annot, all
    self.annot_path = os.path.join(self.db_root, 'annotations')
    self.min_len_video = cfg.get('min_len_video', 0)

    # dataset specifics.
    self._ref_handside = cfg.get('ref_handside', 'right') # all, left, right, interacting
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

    # loading cameras
    with open(os.path.join(self.annot_path,
                           self.annot_subset,
                           'InterHand2.6M_' + self.split + '_camera.json')) as f:
      self.cameras = json.load(f)

    # loading joints
    with open(os.path.join(self.annot_path,
                           self.annot_subset,
                           'InterHand2.6M_' + self.split + '_joint_3d.json')) as f:
      self.joints = json.load(f)

    # load mano params
    with open(os.path.join(self.annot_path,
                           self.annot_subset,
                           'InterHand2.6M_' + self.split + '_MANO_NeuralAnnot.json')) as f:
      self.mano_params = json.load(f)

    # set cache path
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(cache_folder,
        f'Interhand30fps--split_{self.split}--annot_{self.annot_subset}--hand_{self._ref_handside}--minLenFrms{self.min_len_video}--metric_{self._annot_metric}')
    if self.cfg.get('use_mini_dataset', False):
      suffix = '##'.join(self.cfg.mini_dataset[self.split])
      suffix = int(hashlib.sha256(suffix.encode('utf-8')).hexdigest(), 16) % 10**8
      cache_path = cache_path + f'--mini_dataset--{suffix}'
    cache_path = cache_path + '__.pkl'

    # get some video folders
    if use_cache and os.path.exists(cache_path):
      print(f'Loading cache data *** {cache_path}')
      with open(cache_path, 'rb') as f:
        video_folders = pickle.load(f)
    else:
      video_folders = self.get_video_folders()
      if use_cache:
        with open(cache_path, 'wb') as f:
          pickle.dump(video_folders, f)

    # create video objects
    self.videos = list()
    for video_dir, anns in video_folders.items():
      campos,camrot = self.get_video_camposrot(anns[0])
      focal, princpt = self.get_video_focalprincpt(anns[0])
      video_joints = self.get_video_joints3d(anns[0])
      video_mano_params = self.get_video_mano_params(anns[0])
      self.videos.append(InterHandVideo(video_dir=os.path.join(self.db_root,video_dir),
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
                                        supervise_fraction=cfg.get('supervise_fraction',1.0),
                                        add_first_last_frame=cfg.get('add_first_last_frame',True)))
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

    print(f"InterHand2.6M:{self.split}\n \
            num_videos_frames:{self.num_videos_frames}\n \
            num_supervised_videos_frames:{self.num_supervised_videos_frames}\n \
            num_unsupervised_videos_frames:{self.num_unsupervised_videos_frames}")

    # images features
    self.images_features = images_features
    if images_features is not None:
      assert len(images_features)==len(self.list_videos_frames),"Images features are not matched with data."

  def ann2name(self, ann):
    return f"Capture{ann['capture']}/{ann['seq_name']}/cam{ann['camera']}/"

  def is_valid(self, ann):
    # handside
    if self._ref_handside =="all":
      pass
    elif self._ref_handside == ann['hand_type']:
      pass
    else:
      return False
    # joint validity
    if ann['hand_type']=="interacting":
      flag = all([v[0] for v in ann['joint_valid']])
    elif ann['hand_type']=="right":
      flag =  all([v[0] for v in ann['joint_valid']][:21])
    elif ann['hand_type']=="left":
      flag = all([v[0] for v in ann['joint_valid']][-21:])
    return flag

  def get_video_folders(self):
    print(f"Load annotation from {os.path.join(self.annot_path, self.annot_subset)}")
    db = COCO(
          os.path.join(
          self.annot_path,
          self.annot_subset,
          'InterHand2.6M_' + self.split + '_data.json')
          )

    # get all folders
    video_folders = defaultdict(list)
    for aid in db.anns.keys():
      ann = db.anns[aid]
      ann.update(db.loadImgs(ann['image_id'])[0])
      video_dir = os.path.join("images",self.split,self.ann2name(ann))
      if self.is_valid(ann):
        video_folders[video_dir].append(ann)

    # use a subset of videos
    if self.cfg.get('use_mini_dataset', False):
      new_video_folders = dict(filter(lambda elem: elem[0].split(self.split+'/')[-1] in self.cfg.mini_dataset[self.split],video_folders.items()))
      video_folders = new_video_folders

    # threshold length of video
    if self.min_len_video > 0:
      new_video_folders = dict(filter(lambda elem:len(elem[1]) > self.min_len_video, video_folders.items()))
      video_folders = new_video_folders

    # arrange frames of each video
    for video_dir in video_folders:
      video_folders[video_dir] = sorted(video_folders[video_dir], key=lambda elem:int(elem['frame_idx']))

    assert len(video_folders.keys())>0,"Expected some video folders!"
    return video_folders

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



if __name__ == '__main__':
  from easydict import EasyDict
  cfg = {
      "db_root": "misc/datapath/hand-pose",
      "name":"interhand",
      "annot_subset":"all", # "all|human_annot|machine_annot", #INTERHAND
      "ref_handside":"right", # "left|interacting",
      "annot_metric": "m",
      "crop_bbox": false,
      "crop_factor": 1.2,
      "res_size": [256,256], # used if crop_bbox
      "ref_bone": [8,9],
      "root_joint": 0,
      "labelset": ["image","joints3d", "joints3d_rel", "mano_params", "verts3d"], # "avaliable_annotation" in passed batch from dataloader
      #"nseq": 1, # seq of samples 
      #"spacing": 1, #[4,8], #space between samples
      "min_len_video":0, #FILTER SHORT VIDEO
      "supervise_fraction": 1.0, # makes semi-supervised setting as a flag of supervision
      "add_first_last_frame": true,
      "use_mini_dataset": true,
      "mini_dataset":{"test":"Capture0/ROM07_RT_Finger_Occlusions/cam400488/"},#{"train":["train/BB13"]},#"when dict is empty,default json file in './misc fill that",
      "augmentation": {
        "do_scale": false,#"bool same as crop_bbox",
        "same_scale":false ,
        "scale_jitter": 0.0,
        "do_rot": true,
        "same_rot": true,
        "max_rot": 0.3925,
        "do_center": true,
        "same_center": true,
        "center_jitter": 0.1,
        "do_color": true,
        "same_color": false,
          "color": {
            "hue": 0.15,
            "saturation": 0.5,
            "contrast": 0.5,
            "brightness": 0.5,
            "blur_radius": 0.5
          }
        }
      }

  cfg = EasyDict(cfg)
  use_mini_dataset = cfg.get("use_mini_dataset", False)
  mini_dataset = cfg.get('mini_dataset', {})

  if use_mini_dataset:
    if not bool(mini_dataset):
      subset_file = f"misc/{data_config.name}_minimal.json"
      assert os.path.exists(subset_file), f'Not found "{data_file}".'
      with open(subset_file, 'r') as f:
        subset = EasyDict(json.load(f))
        assert split in subset, f"Expected {split} in {subset_file}."
        cfg.mini_dataset = EasyDict({split: subset[split]})

  # image features
  images_features = cfg.get("images_features", None)
  if images_features:
    feature_file = (images_features).format(split=split)
    assert os.path.exists(feature_file), f'Not found "{feature_file}".'
    with open(feature_file, 'rb') as f:
      images_features = pickle.load(f)

  dataset = InterHand(cfg,
                      split='test',
                      mano_root='misc/',
                      images_features=images_features)
