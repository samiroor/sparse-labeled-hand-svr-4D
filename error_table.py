import os
import pickle


filter_fn = lambda k: all(
    ['samples_number' not in k,
     'list' not in k,
     'sub' not in k,
     'woR'not in k,
     'scale_trans_al' not in k])

paper_headers = {
    "methods":"methods",
    "samples_number":"---",
    "f_al_score_5":'F@5',
    "f_al_score_15":'F@15',
    "f_score_5":"---",
    "f_score_15":"---",
    "mesh_auc3d":'AUC_V',
    "xyz_auc3d":'AUC_J',
    "mesh_mean3d":'MPVPE',
    "mesh_aligned_mean3d":"---",
    "mesh_aligned_auc3d":"---",
    "xyz_mean3d" :'MPJPE',
    "xyz_trans_al_mean3d":'MPJPE (Trans-aligned)',
    "xyz_trans_al_auc3d":"---",
    "xyz_procrustes_al_mean3d":'MPJPE (Proc-aligned)',
    "xyz_procrustes_al_auc3d":"---",
    }


def get_csv_file(
    methods_dict,
    ON_SPLIT,
    save_dir,
    ):

  # methods names
  methods = list(methods_dict.keys())

  # header of table
  err_dir = os.path.join(methods_dict[methods[0]], 'errors.pkl')
  with open(err_dir, 'rb') as fp:
    metrics_dict = pickle.load(fp)
    joints_headers = list(filter(filter_fn, [k for k in metrics_dict['joints'].keys()]))
    mesh_headers = list(filter(filter_fn, [k for k in metrics_dict['mesh'].keys()]))
  headers = ['methods', 'samples_number'] + joints_headers + mesh_headers

  # feed to csv file
  os.makedirs(save_dir, exist_ok=True)
  csv_file = os.path.join(save_dir, f'error_table_on_split_{ON_SPLIT}.csv')
  print(f"Saving error table @ {csv_file}")

  with open(csv_file, 'w') as f:
    # headers
    for h in headers:
      f.write(f'{h},')
    f.write('\n')

    for h in headers:
      f.write(f'{paper_headers[h]},')
    f.write('\n')

    # rows
    for method in methods:
      err_dir = os.path.join(methods_dict[method], 'errors.pkl')
      with open(err_dir, 'rb') as fp:
        metrics_dict = pickle.load(fp)
        row = [method]
        row += [metrics_dict['joints']['samples_number']]
        row += [metrics_dict['joints'][k] for k in joints_headers]
        row += [metrics_dict['mesh'][k] for k in mesh_headers]

      for r in row:
        f.write(f'{r},')
      f.write('\n')

