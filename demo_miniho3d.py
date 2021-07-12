import os
import pickle
import argparse


import  error_table, visualize_samples
OUTPUT_DIR = 'outputs_minHo3D_velocity_field_RotationTranslation'


parser = argparse.ArgumentParser(description='creating error tables')
parser.add_argument('--on_split', default='val')
parser.add_argument('--visualize', dest='visualize', action='store_true')

parser.add_argument('--sampler_mode', default='unsupervised')
parser.add_argument('--exp_dir', default='exps_miniHo3d/obman/supFrac_0.0078125_lr5e-5/', help="Experiment Directory & config") #just to have dataset on on-split to vis


if __name__ == '__main__':
  args = parser.parse_args()
  ON_SPLIT = args.on_split

  # methods dict : directory contatins error.pkl with ['joints', 'mesh'] keys and metric values 
  methods_dict = {
    "mesh_predictor(s)":f"exps_miniHo3d/obman/supFrac_0.0078125_lr5e-5/run/test_on_ckpt_best_loss_Setting__batch_size_64_workers_8_datasets__supervised_singlehand_img__split_{ON_SPLIT}_supervise_fraction_0.0078125_eval_mesh_predictor_on_unsupervised_samples",

    "mesh_predictor(s_s)":f"exps_miniHo3d/pairdisentangleflow_obman_pairSupervisedSupervised_supFrac_0.0078125/encoderFeat_contextBase_null_contextHead_null_op_subtract_velocityFieldRootReluH128/run/test_on_ckpt_best_loss_Setting__batch_size_32_workers_8_datasets__pair_supervised_unsupervised_singlehand_img__split_{ON_SPLIT}_supervise_fraction_0.0078125_gap_1,64_eval_mesh_predictor_on_unsupervised_samples",

    "pose_alignment(s_s)":f"exps_miniHo3d/pairdisentangleflow_obman_pairSupervisedSupervised_supFrac_0.0078125/encoderFeat_contextBase_null_contextHead_null_op_subtract_velocityFieldRootReluH128/run/test_on_ckpt_best_loss_Setting__batch_size_32_workers_8_datasets__pair_supervised_unsupervised_singlehand_img__split_{ON_SPLIT}_supervise_fraction_0.0078125_gap_1,64_eval_flow_infer_on_unsupervised_samples",

    "mesh_predictor(s_u)":f"exps_miniHo3d/pairdisentangleflow_obman_pairSupervisedAny_supFrac_0.0078125/s_u_ratio_0.5_allPretrainedSupSup_meshPredictorUpdate_encoderFeat_contextBase_null_contextHead_null_op_subtract_velocityFieldRootReluH128/run/test_on_ckpt_best_loss_Setting__batch_size_32_workers_8_datasets__pair_supervised_unsupervised_singlehand_img__split_{ON_SPLIT}_supervise_fraction_0.0078125_gap_1,64_eval_mesh_predictor_on_unsupervised_samples",

    "pose_alignment(s_u)":f"exps_miniHo3d/pairdisentangleflow_obman_pairSupervisedAny_supFrac_0.0078125/s_u_ratio_0.5_allPretrainedSupSup_meshPredictorUpdate_encoderFeat_contextBase_null_contextHead_null_op_subtract_velocityFieldRootReluH128/run/test_on_ckpt_best_loss_Setting__batch_size_32_workers_8_datasets__pair_supervised_unsupervised_singlehand_img__split_{ON_SPLIT}_supervise_fraction_0.0078125_gap_1,64_eval_flow_infer_on_unsupervised_samples",
    }

  # get error
  error_table.get_csv_file(
      methods_dict,
      ON_SPLIT,
      save_dir=OUTPUT_DIR,
      )

  # get vis
  methods_vis_dict ={k:v for k,v in methods_dict.items()\
      if k in ('mesh_predictor(s_s)', 'pose_alignment(s_u)')}

  if args.visualize:
    visualize_samples.main(
      args,
      methods_dict,
      save_dir=os.path.join(OUTPUT_DIR,f'figs_split_{ON_SPLIT}'),
      )
