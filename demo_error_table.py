import os
import pickle
import argparse

import  error_table


parser = argparse.ArgumentParser(description='creating error tables')

parser.add_argument('--exp_root', default='exps_miniHo3d') #TODO cahnge
parser.add_argument('--output_dir', default='outputs_demo_miniHo3d')#TODO change

parser.add_argument('--baseline_dir', default='obman/supFrac_0.0078125_lr5e-5/run/')
parser.add_argument('--proposed_dir', default='pairdisentangleflow_obman_pairSupervisedAny_supFrac_0.0078125/s_u_ratio_0.5_velocityfieldResflow/run/')


def get_methods_list(args):

  refname = 'test_on_ckpt_{ckpt}_Setting__batch_size_32_workers_8_datasets__sampler_dataset__split_{split}_supervise_fraction_0.0078125_eval_{module}_on_{sampler_mode}_samples'
  exps_list = [
      ['Baseline',                         args.baseline_dir, 'best_loss',                         'mesh_predictor', 'test',  'all'],
      ['Baseline + Alignment Supervision', args.proposed_dir, 'best_mesh_predictor_u__xyz_mean3d', 'mesh_predictor', 'test',  'all'],
      ['Baseline',                         args.baseline_dir, 'best_loss',                         'mesh_predictor', 'test',  'unsupervised'],
      ['Baseline + Alignment Supervision', args.proposed_dir, 'best_mesh_predictor_u__xyz_mean3d', 'mesh_predictor', 'test',  'unsupervised'],
      ['Annotation Propagtion',            args.proposed_dir, 'best_mesh_predictor_u__xyz_mean3d', 'flow_infer',     'test',  'unsupervised'],
      ['Baseline',                         args.baseline_dir, 'best_loss',                         'mesh_predictor', 'train', 'unsupervised'],
      ['Baseline + AlignmentSupervision',  args.proposed_dir, 'best_mesh_predictor_u__xyz_mean3d', 'mesh_predictor', 'train', 'unsupervised'],
      ['Annotation Propagtion',            args.proposed_dir, 'best_mesh_predictor_u__xyz_mean3d', 'flow_infer',     'train', 'unsupervised'],
      ]

  methods_list = []
  for name, _dir, ckpt, module, split, sampler_mode in exps_list:
    exp_dir = os.path.join(args.exps_root, _dir, refname.format(split=split, module=module, sampler_mode=sampler_mode))
    methods_list += [[name, exp_dir, f'split_{split},{sampler_mode}_frames']]

  return methods_list


if __name__ == "__main__":
  args = parser.parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  methods_list = get_methods_list(args)

  error_table.get_csv_file(
      methods_list=methods_list,
      save_file=os.path.join(args.output_dir, f'error_table_{args.exp_root}.csv'),
      )
