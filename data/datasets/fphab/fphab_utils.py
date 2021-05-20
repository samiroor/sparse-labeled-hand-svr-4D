import os
import trimesh
import numpy as np

def load_subjects_infos(data_dir, subjects):
  """ Creates dictionary of subjects informations
    Args:
      - data_dir :(str) root dir of FPHAB
      - subjects :(list) list of subjects
      Returns:
        - subjects_infos : {(subject, action_name, action_seq_idx):nframe}
  """
  print("Load subjects information for FPHAB")

  subjects_infos = {}
  for subject in subjects:
    subject_path = os.path.join(data_dir,
                                "Subjects_info",
                                "{}_info.txt".format(subject))
    with open(subject_path, "r") as fw:
      lines = fw.readlines()
      for line in lines[3:]:
        line = " ".join(line.split())
        action_name, action_seq_idx, nframe = line.strip().split(" ")
        subjects_infos[(subject, action_name, action_seq_idx)]= int(nframe)

  print("Length of subjects_infos is {}".format(len(subjects_infos.keys())))
  return subjects_infos

def load_objects_infos(data_dir):
  """Creates dictionary of objects informations
    Args:
      - data_dir :(str) root dir of FPHAB
      Returns:
        - objects_infos :{subject:{(action_name, action_seq_idx):{frame:(object_name, trans_matrix)}}}
  """
  obj_root = os.path.join(data_dir,"Object_6D_pose_annotation_v1_1")
  subjects = os.listdir(obj_root)
  annots = {}
  for subject in subjects:
    subject_dict = {}
    subj_path = os.path.join(obj_root, subject)
    actions = os.listdir(subj_path)
    for action_name in actions:
      object_name = "_".join(action_name.split("_")[1:])
      action_path = os.path.join(subj_path, action_name)
      seqs = os.listdir(action_path)
      for action_seq_idx in seqs:
        seq_path = os.path.join(action_path,
                                action_seq_idx,
                                "object_pose.txt")
        subject_dict[(action_name, action_seq_idx)]={}
        with open(seq_path, "r") as seq_f:
          raw_lines = seq_f.readlines()
        for raw_line in raw_lines:
          line = raw_line.strip().split(" ")
          frame = int(line[0])
          trans_matrix = np.array(line[1:]).astype(np.float32)
          trans_matrix = trans_matrix.reshape(4, 4).transpose()
          subject_dict[(action_name, action_seq_idx)][frame]= (object_name, trans_matrix)
    annots[subject] = subject_dict
  return annots

def load_objects(data_dir, object_names):
  """ Creates dictionary of objects 
    Args:
      - data_dir :(str) root dir of FPHAB
      Returns:
        - all_models :{obj_name:{"verts":,"faces":}}
  """
  obj_root= os.path.join(data_dir,"Object_models")
  all_models = {}
  for obj_name in object_names:
    obj_path = os.path.join(obj_root,
                            "{}_model".format(obj_name),
                            "{}_model.ply".format(obj_name))
    mesh = trimesh.load(obj_path)
    if obj_name == "juice":
      obj_name = "juice_bottle"
    all_models[obj_name] = {"verts": np.array(mesh.vertices), "faces": np.array(mesh.faces)}
  return all_models

def get_action_split(data_dir, subjects_infos):
  """ Creates train | test lists of sequences
    Args:
      - data_dir :(str) root dir of FPHAB
      Returns:
        - train_seqs : [(subject, action_name, action_seq_idx,nframe)]
        - test_seqs : [(subject, action_name, action_seq_idx,nframe)]
  """
  print("Get (train | test) splits based on actions")
  split_path = os.path.join(data_dir,
                            "data_split_action_recognition.txt")

  train_seqs =[]; test_seqs=[]; test_flag=False
  with open(split_path,"r") as annot_f:
    lines = annot_f.readlines()

  for line in lines[1:]:
    if line.startswith("Test"):
      test_flag = True
      continue
    subject, action_name, action_seq_idx = line.split(" ")[0].split("/")
    action_idx = line.split(" ")[1].strip()
    nframe = subjects_infos[(subject, action_name, action_seq_idx)]
    if test_flag:
      test_seqs.append((subject, action_name, action_seq_idx, nframe))
    else:
      train_seqs.append((subject, action_name, action_seq_idx, nframe))
  print(" Number of train seqs in  action split  {}".format(len(train_seqs)))
  print(" Number of test seqs in action split  {}".format(len(test_seqs)))

  return train_seqs, test_seqs

def get_subject_split(data_dir, subjects_infos):
  """ Creates train | test lists of sequences
    Args:
      - data_dir :(str) root dir of FPHAB
      - subjects_infos
      Returns:
        - train_seqs : [(subject, action_name, action_seq_idx,nframe)]
        - test_seqs : [(subject, action_name, action_seq_idx,nframe)]
  """
  print("Get (train | test) splits based on subjects")
  train_seqs = []; test_seqs = []
  train_subjects = ["Subject_1", "Subject_3", "Subject_4"]
  test_subjects = ["Subject_2", "Subject_5", "Subject_6"]

  for (subject, action_name, action_seq_idx) in subjects_infos.keys():
    nframe = subjects_infos[(subject, action_name, action_seq_idx)]
    if subject in train_subjects:
      train_seqs.append((subject, action_name, action_seq_idx, nframe))
    elif subject in test_subjects:
      test_seqs.append((subject, action_name, action_seq_idx, nframe))
  print(" Number of train seqs in split type of subjects {}".format(len(train_seqs)))
  print(" Number of test seqs in split type of subjects {}".format(len(test_seqs)))
  return train_seqs, test_seqs
