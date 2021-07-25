import os
import pickle
from evaluate_modules import save_pickle, read_pickle



filter_fn = lambda k: all(
    ['samples_number' not in k,
     'list' not in k,
     'sub' not in k,
     'woR'not in k,
     'scale_trans_al' not in k])


paper_headers = {
    "method":"method",
    "split":"split",
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
    "xyz_scale_depth_al_mean3d":'MPJPE (ScaleDepth-aligned)',
    "xyz_scale_depth_al_auc3d":"---",
    "xyz_procrustes_al_mean3d":'MPJPE (Proc-aligned)',
    "xyz_procrustes_al_auc3d":"---",
    }


def get_table_headers(sample_method):
  name, exp_dir, split = sample_method
  error_file = os.path.join(exp_dir, 'errors.pkl')

  metrics_dict = read_pickle(error_file)

  joints_headers = list(filter(filter_fn, [k for k in metrics_dict['joints'].keys()]))
  mesh_headers = list(filter(filter_fn, [k for k in metrics_dict['mesh'].keys()]))

  headers = ['split', 'method', 'samples_number'] + joints_headers + mesh_headers
  return headers


def get_csv_file(
    methods_list,
    save_file,
    ):

  # header of table
  headers = get_table_headers(methods_list[0])

  # feed to csv file
  print(f"Saving error table @ {save_file}")
  with open(save_file, 'w') as f:
    # headers
    for h in headers:
      f.write(f'{h},')
    f.write('\n')

    for h in headers:
      f.write(f'{paper_headers[h]},')
    f.write('\n')

    # rows
    for method_name, exp_dir, split in methods_list:
      error_file = os.path.join(exp_dir, 'errors.pkl')
      print(f'####### Reading {error_file} ########')

      metrics_dict = read_pickle(error_file)

      row = [split, method_name]
      row += [metrics_dict['joints']['samples_number']]
      for h in headers[3:]:
        if h in metrics_dict['joints']:
          row += [metrics_dict['joints'][h]]
        elif h in metrics_dict['mesh']:
          row += [metrics_dict['mesh'][h]]
        else:
          raise ValueError

      for r in row:
        if isinstance(r, str):
          f.write(f'{r},')
        else:
          f.write(f'{r:3.3f},')
      f.write('\n')

