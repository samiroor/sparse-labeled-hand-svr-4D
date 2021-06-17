class Fields(object):
  # typical annotations passed by loader
  images_paths = "images_paths"
  handsides = "handsides"
  masks = "masks"
  images = "images"
  camsintr = "camsintr"

  mano_params = "mano_params"
  mano_shapes = "mano_shapes"
  mano_poses = "mano_poses"
  mano_poses_quat = "mano_poses_quat"

  verts3dcan= "verts3dcan"
  verts3d = "verts3d"
  verts2d = "verts2d"
  faces = "faces"

  joints3dcan = "joints3dcan"
  joints3d = "joints3d"
  joints2d = "joints2d"
  jointsvis = "jointsvis"

  verts3d_rel = "verts3d_rel"
  joints3d_rel = "joints3d_rel"

  verts3d_nrm = "verts3d_nrm"
  joints3d_nrm = "joints3d_nrm"
  joints3d_root = "joints3d_root"
  joints3d_scale = "joints3d_scale"

  joints2d_hms = "joints2d_hms"

  objcorners3d = "objcorners3d"
  objcorners2d = "objcorners2d"
  objverts3d ="objverts3d"
  objverts2d = "objverts2d"
  objvis = "objvis"

  affinetrans = "affinetrans"

  supervised = "supervised" # bool: to be in supervised list or not

  # annotations @ original image before any data processing
  joints2d_prim = "joints2d_prim"
  objfaces_prim = "objfaces_prim"
  objverts2d_prim = "objverts2d_prim"
  objverts3d_prim = "objverts3d_prim"
  objcorners3d_prim = "objcorners3d_prim"
  objcorners2d_prim = "objcorners2d_prim"
  objverts3dcan_prim = "objevertscan_prim"
  objcorners3dcan_prim = "objcorners3dcan_prim"

  # features
  images_features = "images_features" #TODO omit
  encoder_features = "encoder_features"
  encoder_spatial_features = "encoder_spatial_features"
  tail_features = "tail_features"
  encoder_tail_features = "encoder_tail_features"

  # flow loaders
  verts3d_t0 = "verts3d_t0" # verts3d @ t0
  verts3d_t1 = "verts3d_t1" # verts3d @ t1
  t0_time = "t0_time" # t0 = a time
  t1_time = "t1_time" # t1 = a time
  seq_time = "seq_time" # time steps for a given seq
