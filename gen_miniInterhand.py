import os
import json
import pickle
from collections import defaultdict, OrderedDict
from pycocotools.coco import COCO

############ functions ##################

def is_valid(ann, ref_handside='right'):
  # handside validity
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


def ann2video_name(ann):
  return f"Capture{ann['capture']}/{ann['seq_name']}/cam{ann['camera']}/"


def ann2video_wocam(ann):
  return f"Capture{ann['capture']}/{ann['seq_name']}/"


########### setting ##################

db_root = 'misc/datapath/hand-pose/InterHand2.6M/InterHand2.6M_30fps_batch1'
annot_subset = 'all'
split = 'test'#'val' 

save = True
save_dir = 'mini-interhand30fps-v1/annotations'
save_file = 'test_annotations.pkl'#'train_annotations.pkl'
os.makedirs(save_dir, exist_ok=True)

# filter
ref_handside = 'right'
min_len_video = 1000


############# load annotations ###########
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
summary = f'{len(db.anns)} total frames in split {split} from COCO loader \n'

# get all folders filtered by handside
video_folders = defaultdict(list)
video_subjects = set()

for aid in db.anns.keys():
  ann = db.anns[aid]
  ann.update(db.loadImgs(ann['image_id'])[0])

  video_dir = os.path.join("images",split,ann2video_name(ann))
  video_sub = os.path.join("images",split,ann2video_wocam(ann))

  if is_valid(ann, ref_handside):
    video_folders[video_dir].append(ann)
    video_subjects.add(video_sub)


# get videos statistic 
db = None
list_num_frames = list()

for v in video_folders:
  num_frames =  len(video_folders[v])
  list_num_frames.append(num_frames)

minimum_length = min(list_num_frames)
maximum_length = max(list_num_frames)
total_frames = sum(list_num_frames)

summary += 'list_num_frames \n' + '\n'.join(map(str,list_num_frames))

summary += f'\n \
             - total videos frames {total_frames} for handside {ref_handside} in split {split} \n \
             - number of videos {len(list(video_folders.keys()))} \n \
             - maximum length in videos {maximum_length} \n \
             - minimum_length in videos {minimum_length} \n \
             - number of videos with out different cameras {len(video_subjects)}\n'

text_file = open(os.path.join(save_dir,f"orig_DataStatistics_split_{split}_handside_{ref_handside}.txt"), "w")
text_file.write(summary)
text_file.close()


################# Filter short videos #####################
summary = f'Filtered short videos for minimum length {min_len_video} \n'

filtered_video_folders = OrderedDict(
    filter(lambda elem:len(elem[1]) >= min_len_video, video_folders.items())
    )

list_num_frames = list()
filtered_subjects = set()
for v in filtered_video_folders:
  filtered_subjects.add(v.split('/cam')[0])
  num_frames = len(filtered_video_folders[v])
  list_num_frames.append(num_frames)

minimum_length = min(list_num_frames)
maximum_length = max(list_num_frames)
total_frames = sum(list_num_frames)


summary += 'list_num_frames \n' + '\n'.join(map(str,list_num_frames))

summary += f'\n \
             - total videos frames {total_frames} for handside {ref_handside} in split {split} \n \
             - number of videos {len(list(filtered_video_folders.keys()))} \n \
             - maximum length in videos {maximum_length} \n \
             - minimum_length in videos {minimum_length} \n \
             - number of videos with out different cameras {len(filtered_subjects)}\n'

text_file = open(os.path.join(save_dir,f"filtered_w_minimum_length{min_len_video}_orig_DataStatistics_split_{split}_handside_{ref_handside}.txt"), "w")
text_file.write(summary)
text_file.close()


######### save annotations in save file ##################
with open(camera_path, 'rb') as fp:
  cameras = json.load(fp)

with open(mano_params_path, 'rb') as fp:
  mano_params = json.load(fp)

with open(joint_3d_path, 'rb') as fp:
  joints = json.load(fp)


if save:
  with open(os.path.join(save_dir, save_file), 'wb') as fp:
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

  text_file = open(os.path.join(save_dir,save_file[:-4]+"_README.txt"), "w")
  text_file.write(summary)
  text_file.close()
