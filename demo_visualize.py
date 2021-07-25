import os
import pickle
import argparse

import visualize_samples
from demo_error_table import get_methods_list


parser = argparse.ArgumentParser(description='creating error tables')

parser.add_argument('--exp_root', default='exps_miniHo3d') #TODO cahnge
parser.add_argument('--output_dir', default='outputs_demo_miniHo3d')#TODO change

parser.add_argument('--test_split', default='test', help="split for visulaization") #TODO
parser.add_argument('--sampler_mode', default='unsupervised') #TODO
parser.add_argument('--exp_dir', default='exps_miniHo3d/obman/supFrac_0.0078125_lr5e-5/', help="Experiment Directory & config") #TODO

parser.add_argument('--baseline_dir', default='obman/supFrac_0.0078125_lr5e-5/run/')
parser.add_argument('--proposed_dir', default='pairdisentangleflow_obman_pairSupervisedAny_supFrac_0.0078125/s_u_ratio_0.5_velocityfieldResflow/run/')

parser.add_argument('--baseline_ckpt', default='best_loss')
parser.add_argument('--proposed_ckpt', default='best_mesh_predictor_u__xyz_mean3d')


if __name__ == "__main__":
  args = parser.parse_args()
  os.makedirs(args.output_dir, exist_ok=True)

  methods_list = get_methods_list(args)
  methods_list = list(filter( lambda a: a[2]==f'split_{args.test_split}_{args.sampler_mode}_frames', methods_list))

  visualize_samples.main(
        methods_list,
        sampler_mode=args.sampler_mode,
        split=args.test_split,
        exp_dir=args.exp_dir,
        save_dir=os.path.join(args.output_dir,f'figs_{args.exp_root}_on_split_{args.test_split}_{args.sampler_mode}_frames'),
        criteria1='Baseline + Alignment Supervision',
        criteria2='Baseline',
        )
