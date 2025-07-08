import torch
import logging

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_model import QuantModel
from qdiff.optimization.block_recon import block_reconstruction
from qdiff.optimization.layer_recon import layer_reconstruction
from opensora.models.layers.blocks import Attention, MultiHeadCrossAttention
from opensora.models.stdit.stdit import STDiTBlock, STDiT
from opensora.models.stdit.modules import Mlp


logger = logging.getLogger(__name__)


def our_model_reconstruction(model, module, calib_data, config, param_types, opt_target, prefix=""):
    # INFO: due to that the layer_reconstruct and block_reconstruct need to feed in the **quantized_whole_model**
    # while the model is used for recursively conduct reconstruction
    # names = []
    # modules = []
    # for name, module in model.named_children():
        # names.append(name)
        # modules.append(module)

    block_reconstruction(model, module, calib_data, config, param_types, opt_target)
    return
