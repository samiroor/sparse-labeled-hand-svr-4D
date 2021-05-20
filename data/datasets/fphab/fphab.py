import os
import glob
import pickle
import numpy as np

from . import fphab_utils
from .fphab_video import FPHABVideo

from ..videos import Videos
from ..torch_handmodel import Torch_HandModel


class FPHAB(Videos):
  def __init__(self, cfg, split="train", mano_root=None, images_features=None):
    """
    Args:
      cfg (EasyDict): of dataset cfg. Must have the following items:
        - db_root (str): path of dataset.
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
      images_features (dict): {"video_id:frame_id": corresponding features}
    """
    super(FPHAB, self).__init__()
    self.cfg = cfg
    self.split = split

    # get data root; video_folders; object_meshes; hand_model
    self.filter_object = None
    self.db_root = os.path.join(cfg.db_root, "FPHAB")
    self.subjects = ["Subject_1",
                     "Subject_2",
                     "Subject_3",
                     "Subject_4",
                     "Subject_5",
                     "Subject_6"]
    self.all_objects = ["juice",
                        "liquid_soap",
                        "milk",
                        "salt"]
    self.split_mode = cfg.get('split_mode',"actions")
    self.use_objects = cfg.get('use_objects', True)
    self.min_len_video = cfg.get('min_len_video', 0)

    # dataset specifics.
    self._ref_handside = cfg.get('ref_handside', "right")
    self._annot_metric = cfg.get('annot_metric', 'm')
    self._root_joint = cfg.get('root_joint', 0)
    self._ref_bone = cfg.get('ref_bone', [8,9])
    self._crop_bbox = cfg.get('crop_bbox', False)
    self._crop_factor = cfg.get('crop_factor', 1.2)

    resolution_factor = cfg.get('resolution_factor',0.25)
    image_prefix = "Video_files_480" if resolution_factor==0.25 else "Video_files"
    self._image_size = [int(1920 * resolution_factor), int(1080 * resolution_factor)]

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

    self._torch_hand_model = {'right':Torch_HandModel(mano_root,
                                                     use_pca=False,
                                                     use_mean_pose=False,
                                                     ncomps=45,
                                                     ref_side="right",
                                                     center_idx=self._root_joint,
                                                     annot_metric=self._annot_metric,
                                                     use_adpator=True,
                                                     adaptor_path=f"./misc/mano_v1_2/models/fhb_skel_centeridx{cfg.root_joint}.pkl")}

    # get infos
    reorder_idx = np.array([0,
                            1, 6, 7, 8,
                            2, 9, 10, 11,
                            3, 12, 13, 14,
                            4, 15, 16, 17,
                            5, 18, 19, 20])

    self.object_infos = fphab_utils.load_objects_infos(self.db_root)
    self.fhb_objects = fphab_utils.load_objects(self.db_root, object_names=self.all_objects)
    self.subjects_infos = fphab_utils.load_subjects_infos(self.db_root, self.subjects)

    # create video objects
    video_folders = self.get_video_folders()
    self.videos =[FPHABVideo(self.db_root,
                             video_dir,
                             self.get_video_skel_info(video_dir),
                             self.get_video_mano_info(video_dir),
                             self.get_video_object_info(video_dir),
                             fhb_objects=self.fhb_objects,
                             reorder_idx=reorder_idx,
                             resolution_factor=resolution_factor,
                             image_prefix=image_prefix,
                             image_size=self._image_size,
                             annot_metric=self._annot_metric,
                             crop_bbox=self._crop_bbox,
                             crop_factor=self._crop_factor,
                             supervise_fraction=cfg.get('supervise_fraction',1.0),
                             add_first_last_frame=cfg.get('add_first_last_frame',True))  for video_dir in video_folders]

    # list of ids  ['vid_id:frm_id', ...]
    self.list_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._all_frames]
    self.list_supervised_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._supervised_frames]
    self.list_unsupervised_videos_frames = ['{}:{}'.format(i,f) for i, v in enumerate(self.videos) for f in v._unsupervised_frames]

    self.num_videos = len(self.videos)
    self.num_videos_frames = len(self.list_videos_frames)
    self.num_supervised_videos_frames = len(self.list_supervised_videos_frames)
    self.num_unsupervised_videos_frames = len(self.list_unsupervised_videos_frames)

    print(f"FPHAB:{self.split}\n \
            num_videos_frames:{self.num_videos_frames}\n \
            num_supervised_videos_frames:{self.num_supervised_videos_frames}\n \
            num_unsupervised_videos_frames:{self.num_unsupervised_videos_frames}")

    # features
    self.images_features = images_features
    if images_features is not None:
      assert len(images_features)==len(self.list_videos_frames),"Images features are not matched with data."

  def get_video_folders(self):
    # split mode (actions, subjects)
    if self.split_mode == "subjects":
      train_seqs, test_seqs =  fphab_utils.get_subject_split(self.db_root, self.subjects_infos)
    elif self.split_mode == "actions":
      train_seqs, test_seqs =  fphab_utils.get_action_split(self.db_root, self.subjects_infos)
    else:
      raise ValueError(f'Not a valid "{self.split_mode}"')
    # get subjects
    if self.split_mode == "subjects" and self.split == "train":
      subjects = ["Subject_1", "Subject_3", "Subject_4"]
    elif self.split_mode == "subjects" and (self.split =="test" or self.split == "val"):
      subjects = ["Subject_2", "Subject_5", "Subject_6"]
    else:
      subjects = self.subjects

    seqs = train_seqs if self.split == "train" else test_seqs

    # get video dirs
    video_dirs = []
    for (subject, action_name, action_seq_idx, nframe) in seqs:
      if subject in subjects:
        if self.use_objects:
          if subject in self.object_infos and (action_name, action_seq_idx) in self.object_infos[subject]:
            pass
          else:
            continue
      else:
        continue
      video_dirs.append(os.path.join(subject, action_name, action_seq_idx))

    # mini dataset
    if self.cfg.get('use_mini_dataset',False):
      video_dirs = self.cfg.mini_dataset[self.split] #subject/action_name/action_seq_idx

    # threshold length of video
    if self.min_len_video > 0:
      new_video_dirs = [video_dir for video_dir in video_dirs if len(self.get_video_skel_info(video_dir)) > self.min_len_video]
      video_dirs = new_video_dirs

    video_folders = video_dirs
    assert len(video_folders)>0,"Expected some video folders!"
    return video_folders

  def get_video_skel_info(self, video_dir):
    fw = os.path.join(self.db_root,
                      "Hand_pose_annotation_v1",
                      video_dir,
                      "skeleton.txt")
    skel_vals = np.loadtxt(fw)
    if len(skel_vals):
      assert np.all(skel_vals[:, 0]==list(range(skel_vals.shape[0]))), f"row idxs should match frame idx failed at {fw}"
      skel_info = skel_vals[:, 1:].reshape(skel_vals.shape[0],21,-1)
    else:
      # handle sequences of size 0
      skel_info = skel_vals
    return skel_info

  def get_video_mano_info(self, video_dir):
    mano_path = os.path.join(self.db_root,
                             "fhbhands_fits",
                             video_dir,
                             "pkls.pkl")
    with open(mano_path, "rb") as fw:
      mano_info = pickle.load(fw)
    return mano_info

  def get_video_object_info(self, video_dir):
    subject,action_name,action_seq_idx = video_dir.split('/')
    return self.object_infos[subject][(action_name, action_seq_idx)]
