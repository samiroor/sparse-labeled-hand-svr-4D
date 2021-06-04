import os
import glob
import logging
from collections import defaultdict

from . import ho3d_utils
from .ho3d_video import HO3DVideo

from ..videos import Videos
from ..torch_handmodel import Torch_HandModel


class HO3D(Videos):
  def __init__(
      self,
      cfg,
      split="train",
      use_mini_dataset=False,
      mini_dataset={},
      supervise_fraction=1.0,
      add_first_last_frame=True,
      mano_root=None,
      images_features=None,
      **kwargs,
      ):
    """
    Args:
      cfg (EasyDict): of dataset cfg. Must have the following items:
        - db_root (str): path of dataset.
        - name (str) == ho3d
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
    super(HO3D, self).__init__()
    self.split = split
    self.use_mini_dataset = use_mini_dataset
    self.mini_dataset = mini_dataset
    self.supervise_fraction = supervise_fraction
    self.add_first_last_frame = add_first_last_frame
    self.images_features = images_features


    # get data root; video_folders; object_meshes; hand_model
    self.db_root = os.path.join(cfg.db_root, "HO3D", "HO3D_v2")
    self.obj_meshes = ho3d_utils.load_objects(os.path.join(self.db_root, "modelsprocess"))
    self.min_len_video = cfg.get('min_len_video', 0)

    # dataset specifics.
    self._ref_handside = cfg.get('ref_handside', "right")
    self._annot_metric = cfg.get('annot_metric', 'm')
    self._root_joint = cfg.get('root_joint', 0)
    self._ref_bone = cfg.get('ref_bone', [8,9])
    self._crop_bbox = cfg.get('crop_bbox', False)
    self._crop_factor = cfg.get('crop_factor', 1.2)

    self._image_size = [640, 480]
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
                                                      use_mean_pose=False,
                                                      ncomps=45,
                                                      ref_side="right",
                                                      center_idx=self._root_joint,
                                                      annot_metric=self._annot_metric)}
    # create video objects
    video_folders = self.get_video_folders()
    self.videos = [HO3DVideo(video_dir=video_dir,
                             anns=anns,
                             obj_meshes=self.obj_meshes,
                             image_size=self._image_size,
                             annot_metric=self._annot_metric,
                             crop_bbox=self._crop_bbox,
                             crop_factor=self._crop_factor,
                             supervise_fraction=self.supervise_fraction,
                             add_first_last_frame=self.add_first_last_frame) for
                             video_dir,anns in video_folders.items()]

    # list of ids ['video_id:frame_id', ...]
    self.list_videos_frames = ['{}:{}'.format(i,f) for i,v in enumerate(self.videos) for f in v._all_frames]
    self.list_supervised_videos_frames = ['{}:{}'.format(i,f) for i,v in enumerate(self.videos) for f in v._supervised_frames]
    self.list_unsupervised_videos_frames = ['{}:{}'.format(i,f) for i,v in enumerate(self.videos) for f in v._unsupervised_frames]

    self.num_videos = len(self.videos)
    self.num_videos_frames = len(self.list_videos_frames)
    self.num_supervised_videos_frames = len(self.list_supervised_videos_frames)
    self.num_unsupervised_videos_frames = len(self.list_unsupervised_videos_frames)

    logging.info(f"HO3D:{self.split}\n \
                   num_videos_frames:{self.num_videos_frames}\n \
                   num_supervised_videos_frames:{self.num_supervised_videos_frames}\n \
                   num_unsupervised_videos_frames:{self.num_unsupervised_videos_frames}")

    # images features
    if self.images_features is not None:
      assert len(self.images_features)==len(self.list_videos_frames),"Images features are not matched with data."

  def get_video_folders(self):
    # mini dataset [train/seq]
    if self.use_mini_dataset:
      video_dirs = [os.path.join(self.db_root, video_dir) for video_dir in self.mini_dataset[self.split]]

    # get all dirs
    elif self.split=="train" or self.split == "trainval":
      video_dirs = list(sorted(filter(lambda x:os.path.isdir(x), glob.glob(os.path.join(self.db_root,'train/*')))))
      if self.split == "train":
        video_dirs.remove(os.path.join(self.db_root,'train','MC2'))
    elif self.split == "val":
      video_dirs = [os.path.join(self.db_root,'train','MC2')]
    elif self.split == "test":
      # gt annotations of test set  are not available; 
      # results should be submitted to server! 
      video_dirs = list()
      with open(os.path.join(self.db_root, 'evaluation.txt')) as f:
        for line in f:
          video_dir = os.path.join(self.db_root, 'evaluation', line.split('/')[0])
          if video_dir not in video_dirs:
            video_dirs.append(video_dir)
    else:
      raise ValueError("Not a valid dataset split!")

    # get folders
    video_folders = defaultdict(list)
    for video_dir in video_dirs:
      num_frames = len(glob.glob(os.path.join(video_dir,'rgb/*.png')))
      anns = [ho3d_utils.load_pickle_data(os.path.join(video_dir, 'meta', '{:04d}.pkl'.format(frame)))
               for frame in range(num_frames)]
      video_folders[video_dir] = anns


    # threshold length of video
    if self.min_len_video > 0:
      new_video_folders = dict(filter(lambda elem:len(elem[1]) > self.min_len_video, video_folders.items()))
      video_folders = new_video_folders

    assert len(video_folders.keys())>0,"Expected some video folders!"
    return video_folders
