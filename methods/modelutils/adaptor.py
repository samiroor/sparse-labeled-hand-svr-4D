import torch
import torch.nn as nn


from manopth.manolayer import ManoLayer


class ManoAdaptor(torch.nn.Module):
    def __init__(
        self,
        mano_layer=None,
        load_path=None,
        njoints=21,
        nverts=778,
        ):
        super().__init__()
        self.adaptor = torch.nn.Linear(nverts, njoints, bias=False)
        if load_path is not None:
            with open(load_path, "rb") as p_f:
                exp_data = pickle.load(p_f)
                weights = exp_data["adaptor"]
            regressor = torch.Tensor(weights)
            self.register_buffer("J_regressor", regressor)
        else:
            if mano_layer is None:
                mano_layer = ManoLayer(
      ncomps=45,
      use_pca=False,
      side='right',
      center_idx=0,
      flat_hand_mean=False,
      mano_root='misc/mano_v1_2/models',
      )


            regressor = mano_layer._buffers["th_J_regressor"]
            tip_reg = regressor.new_zeros(5, regressor.shape[1])
            tip_reg[0, 745] = 1
            tip_reg[1, 317] = 1
            tip_reg[2, 444] = 1
            tip_reg[3, 556] = 1
            tip_reg[4, 673] = 1
            reordered_reg = torch.cat([regressor, tip_reg])[
                [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
            ]
            self.register_buffer("J_regressor", reordered_reg)
        self.adaptor.weight.data = self.J_regressor

    def forward(self, inp):
        fix_idxs = [0, 4, 8, 12, 16, 20]
        for idx in fix_idxs:
            self.adaptor.weight.data[idx] = self.J_regressor[idx]
        return self.adaptor(inp.transpose(2, 1)), self.adaptor.weight - self.J_regressor


def rec_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        rec_freeze(child)

