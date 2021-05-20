import os
import json
import pickle
from collections import defaultdict, OrderedDict
from pycocotools.coco import COCO

# functions
def is_valid(ann, ref_handside='right'):
  # handside
  if ref_handside =="all":
    pass
  elif ref_handside == ann['hand_type']:
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

def ann2name(ann):
  return f"Capture{ann['capture']}/{ann['seq_name']}/cam{ann['camera']}/"

def ann2subject(ann):
  return f"Capture{ann['capture']}/{ann['seq_name']}/"


########### setting ####################
db_root = 'misc/datapath/hand-pose/InterHand2.6M/InterHand2.6M_30fps_batch1'
split = 'val' #'test'
annot_subset = 'all'
ref_handside = 'right'
min_len_video = 1000

save = True
save_dir = 'mini-interhand30fps-v1/annotations'
filename = 'train_annotations.pkl' #'test_annotations.pkl'
os.makedirs(save_dir, exist_ok=True)


# load annotations
annot_path = os.path.join(
    db_root,
    'annotations',
    annot_subset,
    f'InterHand2.6M_{split}_data.json')
camera_path = os.path.join(
    db_root,
    'annotations',
    annot_subset,
    f'InterHand2.6M_{split}_camera.json')
joint_3d_path = os.path.join(
    db_root,
    'annotations',
    annot_subset,
    f'InterHand2.6M_{split}_joint_3d.json')
mano_params_path = os.path.join(
    db_root,
    'annotations',
    annot_subset,
    f'InterHand2.6M_{split}_MANO_NeuralAnnot.json')


db = COCO(annot_path)
summary = f'{len(db.anns)} total frames in split {split} \n'

# get all folders
video_folders = defaultdict(list)
video_subjects = set()
for aid in db.anns.keys():
  ann = db.anns[aid]
  ann.update(db.loadImgs(ann['image_id'])[0])
  video_dir = os.path.join("images",split,ann2name(ann))
  video_sub = os.path.join("images",split,ann2subject(ann))

  if is_valid(ann, ref_handside):
    video_folders[video_dir].append(ann)
    video_subjects.add(video_sub)

# filter short videos
db = None
total_frames = 0
max_video_len = 0
for v in video_folders:
  num_frames =  len(video_folders[v])
  total_frames += num_frames
  if num_frames > max_video_len:
    max_video_len = num_frames

summary += f'{total_frames} total frames \n \
             for handside {ref_handside} in split {split}-{len(list(video_folders.keys()))} videos \n \
             - maximum len video {max_video_len} frames-number of subjects {len(video_subjects)}\n'

filtered_video_folders = OrderedDict(filter(lambda elem:len(elem[1]) >= min_len_video, video_folders.items()))
filtered_subjects = set()

total_frames = 0
for v in filtered_video_folders:
  filtered_subjects.add(v.split('/cam')[0])
  total_frames += len(filtered_video_folders[v])

summary += f'{total_frames} total frames \n \
             for handside {ref_handside} in split {split} with minimum length {min_len_video} frames\n \
             -{len(list(filtered_video_folders.keys()))} videos - number of subjects {len(filtered_subjects)} \n'


with open(camera_path, 'rb') as fp:
  cameras = json.load(fp)

with open(mano_params_path, 'rb') as fp:
  mano_params = json.load(fp)

with open(joint_3d_path, 'rb') as fp:
  joints = json.load(fp)

if save:
  with open(os.path.join(save_dir, filename), 'wb') as fp:
    pickle.dump({
      'video_folders':filtered_video_folders,
      'video_names':list(filtered_video_folders.keys()),
      'cameras':cameras,
      'joints':joints,
      'mano_params':mano_params,
      'min_len_video':min_len_video,
      'total_videos':len(list(filtered_video_folders.keys())),
      'total_subjects':len(filtered_subjects),
      'total_frames':total_frames,
      'annot_subset':annot_subset,
      'ref_handside':ref_handside,
      'read_from_split':split,
      }, fp)

  text_file = open(os.path.join(save_dir,filename[:-4]+"_README.txt"), "w")
  text_file.write(summary)
  text_file.close()
