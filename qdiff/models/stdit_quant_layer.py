import logging
import torch
from qdiff.quantizer.base_quantizer import WeightQuantizer, ActQuantizer, StraightThrough, round_ste
from qdiff.models.quant_layer import QuantLayer, find_interval
from omegaconf import ListConfig
import copy
import torch.nn as nn
logger = logging.getLogger(__name__)

'''
Utility QuantLayers for STDiT temporal/spatial attn layer linears
'''
    
class QuantSpatialAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''weight_quant_params_res = copy.deepcopy(self.weight_quant_params)
        weight_quant_params_res.n_bits = 1
        weight_quant_params_res.mixed_precision = None
        self.weight_quantizer_res = WeightQuantizer(weight_quant_params_res)'''

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_temporal_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == S

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,T*S,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*T,S,C])

        if self.weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                E = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB.weight.dtype)
                lora_weight = self.loraB(self.loraA(E))
                lora_weight = lora_weight.T
                E_out = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB_out.weight.dtype)
                lora_weight_out = self.loraB_out(self.loraA_out(E_out))
                lora_weight_out = lora_weight_out.T
                if self.smooth_quant:
                    # during the weight init stage
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                    self.weight_quantizer.cur_timestep_id = cur_timerange_id
                    weight = self.weight_quantizer(self.weight * channel_wise_scale + lora_weight)
                else:
                    weight = self.weight_quantizer(self.weight + lora_weight)
                weight = weight + lora_weight_out
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.weight * channel_wise_scale
            else:
                weight = self.weight
            bias = self.bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        # import ipdb; ipdb.set_trace()
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        if torch.isnan(out).any():
            logging.info('nan exist in the activation')
            import ipdb; ipdb.set_trace()

        return out

class QuantTemporalAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        T = self.act_quant_params['n_temporal_token']
        self.mask = nn.Parameter(torch.ones([1, T, 1]))

    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # check the n_spatial/temporal_token num in act_quant_config is True
        BS = input.shape[0]//self.act_quant_params['n_spatial_token']
        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]
        assert input.shape[1] == T

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            input = input.reshape([BS,S*T,C])
            input = self.act_quantizer(input)
            # convert back
            input = input.reshape([BS*S,T,C])

        if self.weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                E = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB.weight.dtype)
                lora_weight = self.loraB(self.loraA(E))
                lora_weight = lora_weight.T
                E_out = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB_out.weight.dtype)
                lora_weight_out = self.loraB_out(self.loraA_out(E_out))
                lora_weight_out = lora_weight_out.T
                if self.smooth_quant:
                    # during the weight init stage
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                    self.weight_quantizer.cur_timestep_id = cur_timerange_id
                    weight = self.weight_quantizer(self.weight * channel_wise_scale + lora_weight)
                else:
                    weight = self.weight_quantizer(self.weight + lora_weight)
                weight = weight + lora_weight_out
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.weight * channel_wise_scale
            else:
                weight = self.weight
            bias = self.bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        if self.weight_quant:
            out_lora = self.fwd_func(input, lora_weight_out, **self.fwd_kwargs)
            out_lora = out_lora * self.mask
            out = out + out_lora
        out = self.activation_function(out)

        if torch.isnan(out).any():
            logging.info('nan exist in the activation')
            import ipdb; ipdb.set_trace()
            
        return out

class QuantCrossAttnLinear(QuantLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: new forward, cleaner
    def forward(self, input: torch.Tensor, scale: float = 1.0, split: int = 0):
        # Need to handle both Q & KV
        # Q_Linear: [BS, T*S, C]
        # KV_Linear: [1, BS*n_prompt, C]

        T = self.act_quant_params['n_temporal_token']
        S = self.act_quant_params['n_spatial_token']
        C = input.shape[2]

        if input.shape[1] == T*S:
            layer_type = "q"
            BS = input.shape[0]
        elif input.shape[0] == 1:
            layer_type = "kv"
            BS = input.shape[1]//self.act_quant_params['n_prompt']
            n_prompt = self.act_quant_params['n_prompt']
        else:
            print('illegeal shape.')
            # import ipdb; ipdb.set_trace()

        if self.smooth_quant:
            cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)

            if isinstance(self.smooth_quant_alpha, (list, ListConfig)):
                alpha = self.smooth_quant_alpha[cur_timerange_id]
            else:
                alpha = self.smooth_quant_alpha

            if self.channel_wise_scale_type == "dynamic":
                channel_wise_scale = input.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)
            elif "momentum" in self.channel_wise_scale_type:
                if self.smooth_quant_running_stat:
                    cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                    if self.act_quantizer.act_scale is None:
                        self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                    if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                        self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                    else:
                        self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)
                else:
                    assert self.act_quantizer.act_scale[cur_timerange_id] is not None
                    assert self.act_quantizer.act_scale[cur_timerange_id].mean() != 0
                    if (self.act_quantizer.act_scale[cur_timerange_id] == 0).sum() != 0:
                        zero_mask = self.act_quantizer.act_scale[cur_timerange_id] == 0
                        eps = 1.e-5
                        self.act_quantizer.act_scale[cur_timerange_id][zero_mask] = eps
                        logging.info('act_scale containing zeros, replacing with {}'.format(eps))

                channel_wise_scale = self.act_quantizer.act_scale[cur_timerange_id].pow(alpha) / self.weight.abs().max(dim=0)[0].pow(1 - alpha)

            else:
                raise NotImplementedError
            input = input / channel_wise_scale
        else:
            if not hasattr(self, 'timerange'):
                cur_timerange_id = 0
            else:
                cur_timerange_id = find_interval(self.timerange, self.cur_timestep_id)
            if getattr(self, "smooth_quant_running_stat", False) and "momentum" in self.channel_wise_scale_type:
                cur_act_scale = input.abs().max(dim=-2)[0].mean(dim=0, keepdim=True)
                if self.act_quantizer.act_scale is None:
                    self.act_quantizer.act_scale = torch.zeros([self.timerange_num, *cur_act_scale.shape]).to(input)
                if self.act_quantizer.act_scale[cur_timerange_id].abs().mean()==0:
                    self.act_quantizer.act_scale[cur_timerange_id] = cur_act_scale
                else:
                    self.act_quantizer.act_scale[cur_timerange_id] = self.act_quantizer.act_scale[cur_timerange_id] * self.smooth_quant_momentum + cur_act_scale * (1 - self.smooth_quant_momentum)

        if not self.disable_act_quant and self.act_quant:
            # convert the dim into [bs, n_token, c]
            if layer_type == 'q':
                input = self.act_quantizer(input)
            elif layer_type == 'kv':
                # INFO: when mask_select=True
                # it only supports dynamic quant
                if not self.act_quant_params.get('dynamic',False):
                    if self.act_quant_params.per_group is False:  # no need to reshape for tensor-wise quant
                        input = self.act_quantizer(input)
                    else:
                        input = input.reshape([BS,n_prompt,C])
                        input = self.act_quantizer(input)
                        input = input.reshape([1,BS*n_prompt,C])
                else:
                    # directly assign N_batch*prompt quant_params for each token
                    input = self.act_quantizer(input)

        if self.weight_quant:
            if self.split != 0:
                weight_0 = self.weight_quantizer(self.weight[:, :self.split, ...])
                weight_1 = self.weight_quantizer_0(self.weight[:, self.split:, ...])
                weight = torch.cat([weight_0, weight_1], dim=1)
            else:
                E = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB.weight.dtype)
                lora_weight = self.loraB(self.loraA(E))
                lora_weight = lora_weight.T
                E_out = torch.eye(self.weight.shape[1], device=input.device).to(self.loraB_out.weight.dtype)
                lora_weight_out = self.loraB_out(self.loraA_out(E_out))
                lora_weight_out = lora_weight_out.T
                if self.smooth_quant:
                    # during the weight init stage
                    if self.weight_quantizer.timestep_wise is None: # reinit the weight_quantizer
                        self.weight_quantizer.timestep_wise = True
                        self.weight_quantizer.n_timestep = len(self.timerange)
                    self.weight_quantizer.cur_timestep_id = cur_timerange_id
                    weight = self.weight_quantizer(self.weight * channel_wise_scale + lora_weight)
                else:
                    weight = self.weight_quantizer(self.weight + lora_weight)
                weight = weight + lora_weight_out
            bias = self.bias
        else:
            if self.smooth_quant:
                weight = self.weight * channel_wise_scale
            else:
                weight = self.weight
            bias = self.bias

        if weight.dtype == torch.float32 and input.dtype == torch.float16:
            weight = weight.to(torch.float16)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)

        if torch.isnan(out).any():
            logging.info('nan exist in the activation')
            import ipdb; ipdb.set_trace()

        return out


