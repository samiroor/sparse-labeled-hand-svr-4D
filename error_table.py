import os
import pickle
from evaluate_modules import save_pickle, read_pickle



table_headers =[
    "Evaluated on",
    "Model",
    "samples_number",
    "f_al_score_5",
    "f_al_score_15",
    #"mesh_auc3d",
    "mesh_mean3d",
    "mesh_aligned_mean3d",
    #"xyz_auc3d",
    "xyz_mean3d",
    "xyz_trans_al_mean3d",
    "xyz_scale_depth_al_mean3d",
    "xyz_procrustes_al_mean3d",
    ]


paper_headers = {
    "Evaluated on":"Evaluated on",
    "Model":"Model",
    "samples_number":"---",
    "f_al_score_5":'F@5[mm]',
    "f_al_score_15":'F@15[mm]',
    "f_score_5":"---",
    "f_score_15":"---",
    "mesh_auc3d":'AUC_V',
    "xyz_auc3d":'AUC_J',
    "mesh_mean3d":'MPVPE [mm]',
    "mesh_aligned_mean3d":'MPVPE (P)[mm]',
    "mesh_aligned_auc3d":"---",
    "xyz_mean3d" :'MPJPE [mm]',
    "xyz_trans_al_mean3d":'MPJPE (T)[mm]',
    "xyz_trans_al_auc3d":"---",
    "xyz_scale_depth_al_mean3d":'MPJPE (SD)[mm]',
    "xyz_scale_depth_al_auc3d":"---",
    "xyz_procrustes_al_mean3d":'MPJPE (P)[mm]',
    "xyz_procrustes_al_auc3d":"---",
    }



def get_csv_file(
    methods_list,
    save_file,
    ):

  # header of table
  headers = table_headers

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

