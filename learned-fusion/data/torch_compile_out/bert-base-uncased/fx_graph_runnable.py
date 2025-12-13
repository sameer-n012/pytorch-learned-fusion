
import os
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/home/ec2-user/final/pytorch-learned-fusion/learned-fusion/_torch_cache'
os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TORCH_LOGS'] = '+inductor,graph,graph_code,aot_graphs,output_code'
os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '1'
os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
os.environ['TORCHINDUCTOR_SAVE_OPERATORS'] = '1'
os.environ['TRITON_SAVE_TTIR'] = '1'
os.environ['TORCHINDUCTOR_FX_COMPILE_MODE'] = 'SERIALIZE'
os.environ['TRITON_CACHE_DIR'] = 'data/torch_compile_out/bert-base-uncased/triton_kernels'
os.environ['TORCH_INDUCTOR_LOG_DIR'] = 'data/torch_compile_out/bert-base-uncased/logs'
os.environ['MY_TORCH_MODEL_OUTPUT_DIR'] = 'data/torch_compile_out/bert-base-uncased'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.replay_side_effects = True
torch._dynamo.config.side_effect_replay_policy = 'silent'
torch._dynamo.config.specialize_int = False
torch._dynamo.config.specialize_float = False
torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.automatic_dynamic_shapes = True
torch._dynamo.config.capture_scalar_outputs = False
torch._dynamo.config.capture_dynamic_output_shape_ops = False
torch._dynamo.config.prefer_deferred_runtime_asserts_over_guards = False
torch._dynamo.config.do_not_emit_runtime_asserts = False
torch._dynamo.config.allow_rnn = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._inductor.config.trace.graph_diagram = True
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.debug_partitioner = True
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None




# torch version: 2.10.0a0+gite9d6fa5
# torch cuda version: 12.6
# torch git version: e9d6fa5266a35e7d1969d8d9fb0416da65e82f1e


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA A10G : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203):
        slice_1 = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 8)
        expand = torch.ops.aten.expand.default(slice_1, [1, 8]);  slice_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 8)
        embedding = torch.ops.aten.embedding.default(primals_4, primals_1, 0);  primals_4 = None
        embedding_1 = torch.ops.aten.embedding.default(primals_5, expand);  primals_5 = expand = None
        add = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        embedding_2 = torch.ops.aten.embedding.default(primals_6, slice_2);  primals_6 = slice_2 = None
        add_1 = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_2 = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3 = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_9, 1);  primals_9 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [1, 1, 8, 8]);  unsqueeze_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(expand_1, torch.float32);  expand_1 = None
        full_default = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1 = torch.ops.aten.sub.Tensor(full_default, convert_element_type);  full_default = convert_element_type = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1 = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(convert_element_type_1, full_default_1, sub_1);  convert_element_type_1 = full_default_1 = sub_1 = None
        view = torch.ops.aten.view.default(add_3, [8, 768])
        permute = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm = torch.ops.aten.addmm.default(primals_11, view, permute);  primals_11 = permute = None
        view_1 = torch.ops.aten.view.default(addmm, [1, 8, 768]);  addmm = None
        view_2 = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1 = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_2 = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1 = torch.ops.aten.addmm.default(primals_13, view, permute_2);  primals_13 = permute_2 = None
        view_4 = torch.ops.aten.view.default(addmm_1, [1, 8, 768]);  addmm_1 = None
        view_5 = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
        permute_3 = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_4 = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_2 = torch.ops.aten.addmm.default(primals_15, view, permute_4);  primals_15 = permute_4 = None
        view_7 = torch.ops.aten.view.default(addmm_2, [1, 8, 768]);  addmm_2 = None
        view_8 = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
        permute_5 = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_flash_attention_for_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_1, permute_3, permute_5, attn_mask = where)
        getitem_2 = _scaled_dot_product_flash_attention_for_cpu[0]
        getitem_3 = _scaled_dot_product_flash_attention_for_cpu[1];  _scaled_dot_product_flash_attention_for_cpu = None
        permute_6 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])
        view_9 = torch.ops.aten.view.default(permute_6, [1, 8, 768]);  permute_6 = None
        view_10 = torch.ops.aten.view.default(view_9, [8, 768]);  view_9 = None
        permute_7 = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_3 = torch.ops.aten.addmm.default(primals_17, view_10, permute_7);  primals_17 = view_10 = permute_7 = None
        view_11 = torch.ops.aten.view.default(addmm_3, [1, 8, 768]);  addmm_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_1[0]
        getitem_5 = var_mean_1[1];  var_mean_1 = None
        add_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2 = torch.ops.aten.sub.Tensor(add_4, getitem_5);  add_4 = getitem_5 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, primals_18)
        add_6 = torch.ops.aten.add.Tensor(mul_3, primals_19);  mul_3 = primals_19 = None
        view_12 = torch.ops.aten.view.default(add_6, [8, 768])
        permute_8 = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_4 = torch.ops.aten.addmm.default(primals_21, view_12, permute_8);  primals_21 = permute_8 = None
        view_13 = torch.ops.aten.view.default(addmm_4, [1, 8, 3072])
        mul_4 = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5 = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6 = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        view_14 = torch.ops.aten.view.default(mul_6, [8, 3072]);  mul_6 = None
        permute_9 = torch.ops.aten.permute.default(primals_22, [1, 0])
        addmm_5 = torch.ops.aten.addmm.default(primals_23, view_14, permute_9);  primals_23 = permute_9 = None
        view_15 = torch.ops.aten.view.default(addmm_5, [1, 8, 768]);  addmm_5 = None
        add_8 = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_2[0]
        getitem_7 = var_mean_2[1];  var_mean_2 = None
        add_9 = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3 = torch.ops.aten.sub.Tensor(add_8, getitem_7);  add_8 = getitem_7 = None
        mul_7 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8 = torch.ops.aten.mul.Tensor(mul_7, primals_24)
        add_10 = torch.ops.aten.add.Tensor(mul_8, primals_25);  mul_8 = primals_25 = None
        view_16 = torch.ops.aten.view.default(add_10, [8, 768])
        permute_10 = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_6 = torch.ops.aten.addmm.default(primals_27, view_16, permute_10);  primals_27 = permute_10 = None
        view_17 = torch.ops.aten.view.default(addmm_6, [1, 8, 768]);  addmm_6 = None
        view_18 = torch.ops.aten.view.default(view_17, [1, -1, 12, 64]);  view_17 = None
        permute_11 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        permute_12 = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_7 = torch.ops.aten.addmm.default(primals_29, view_16, permute_12);  primals_29 = permute_12 = None
        view_20 = torch.ops.aten.view.default(addmm_7, [1, 8, 768]);  addmm_7 = None
        view_21 = torch.ops.aten.view.default(view_20, [1, -1, 12, 64]);  view_20 = None
        permute_13 = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        permute_14 = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_8 = torch.ops.aten.addmm.default(primals_31, view_16, permute_14);  primals_31 = permute_14 = None
        view_23 = torch.ops.aten.view.default(addmm_8, [1, 8, 768]);  addmm_8 = None
        view_24 = torch.ops.aten.view.default(view_23, [1, -1, 12, 64]);  view_23 = None
        permute_15 = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        _scaled_dot_product_flash_attention_for_cpu_1 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_11, permute_13, permute_15, attn_mask = where)
        getitem_8 = _scaled_dot_product_flash_attention_for_cpu_1[0]
        getitem_9 = _scaled_dot_product_flash_attention_for_cpu_1[1];  _scaled_dot_product_flash_attention_for_cpu_1 = None
        permute_16 = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3])
        view_25 = torch.ops.aten.view.default(permute_16, [1, 8, 768]);  permute_16 = None
        view_26 = torch.ops.aten.view.default(view_25, [8, 768]);  view_25 = None
        permute_17 = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_9 = torch.ops.aten.addmm.default(primals_33, view_26, permute_17);  primals_33 = view_26 = permute_17 = None
        view_27 = torch.ops.aten.view.default(addmm_9, [1, 8, 768]);  addmm_9 = None
        add_11 = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_3[0]
        getitem_11 = var_mean_3[1];  var_mean_3 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4 = torch.ops.aten.sub.Tensor(add_11, getitem_11);  add_11 = getitem_11 = None
        mul_9 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10 = torch.ops.aten.mul.Tensor(mul_9, primals_34)
        add_13 = torch.ops.aten.add.Tensor(mul_10, primals_35);  mul_10 = primals_35 = None
        view_28 = torch.ops.aten.view.default(add_13, [8, 768])
        permute_18 = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_10 = torch.ops.aten.addmm.default(primals_37, view_28, permute_18);  primals_37 = permute_18 = None
        view_29 = torch.ops.aten.view.default(addmm_10, [1, 8, 3072])
        mul_11 = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12 = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1 = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        view_30 = torch.ops.aten.view.default(mul_13, [8, 3072]);  mul_13 = None
        permute_19 = torch.ops.aten.permute.default(primals_38, [1, 0])
        addmm_11 = torch.ops.aten.addmm.default(primals_39, view_30, permute_19);  primals_39 = permute_19 = None
        view_31 = torch.ops.aten.view.default(addmm_11, [1, 8, 768]);  addmm_11 = None
        add_15 = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_4[0]
        getitem_13 = var_mean_4[1];  var_mean_4 = None
        add_16 = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_15, getitem_13);  add_15 = getitem_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15 = torch.ops.aten.mul.Tensor(mul_14, primals_40)
        add_17 = torch.ops.aten.add.Tensor(mul_15, primals_41);  mul_15 = primals_41 = None
        view_32 = torch.ops.aten.view.default(add_17, [8, 768])
        permute_20 = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_12 = torch.ops.aten.addmm.default(primals_43, view_32, permute_20);  primals_43 = permute_20 = None
        view_33 = torch.ops.aten.view.default(addmm_12, [1, 8, 768]);  addmm_12 = None
        view_34 = torch.ops.aten.view.default(view_33, [1, -1, 12, 64]);  view_33 = None
        permute_21 = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        permute_22 = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_13 = torch.ops.aten.addmm.default(primals_45, view_32, permute_22);  primals_45 = permute_22 = None
        view_36 = torch.ops.aten.view.default(addmm_13, [1, 8, 768]);  addmm_13 = None
        view_37 = torch.ops.aten.view.default(view_36, [1, -1, 12, 64]);  view_36 = None
        permute_23 = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        permute_24 = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_14 = torch.ops.aten.addmm.default(primals_47, view_32, permute_24);  primals_47 = permute_24 = None
        view_39 = torch.ops.aten.view.default(addmm_14, [1, 8, 768]);  addmm_14 = None
        view_40 = torch.ops.aten.view.default(view_39, [1, -1, 12, 64]);  view_39 = None
        permute_25 = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        _scaled_dot_product_flash_attention_for_cpu_2 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_21, permute_23, permute_25, attn_mask = where)
        getitem_14 = _scaled_dot_product_flash_attention_for_cpu_2[0]
        getitem_15 = _scaled_dot_product_flash_attention_for_cpu_2[1];  _scaled_dot_product_flash_attention_for_cpu_2 = None
        permute_26 = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3])
        view_41 = torch.ops.aten.view.default(permute_26, [1, 8, 768]);  permute_26 = None
        view_42 = torch.ops.aten.view.default(view_41, [8, 768]);  view_41 = None
        permute_27 = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_15 = torch.ops.aten.addmm.default(primals_49, view_42, permute_27);  primals_49 = view_42 = permute_27 = None
        view_43 = torch.ops.aten.view.default(addmm_15, [1, 8, 768]);  addmm_15 = None
        add_18 = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_16 = var_mean_5[0]
        getitem_17 = var_mean_5[1];  var_mean_5 = None
        add_19 = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_18, getitem_17);  add_18 = getitem_17 = None
        mul_16 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
        mul_17 = torch.ops.aten.mul.Tensor(mul_16, primals_50)
        add_20 = torch.ops.aten.add.Tensor(mul_17, primals_51);  mul_17 = primals_51 = None
        view_44 = torch.ops.aten.view.default(add_20, [8, 768])
        permute_28 = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_16 = torch.ops.aten.addmm.default(primals_53, view_44, permute_28);  primals_53 = permute_28 = None
        view_45 = torch.ops.aten.view.default(addmm_16, [1, 8, 3072])
        mul_18 = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19 = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2 = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        view_46 = torch.ops.aten.view.default(mul_20, [8, 3072]);  mul_20 = None
        permute_29 = torch.ops.aten.permute.default(primals_54, [1, 0])
        addmm_17 = torch.ops.aten.addmm.default(primals_55, view_46, permute_29);  primals_55 = permute_29 = None
        view_47 = torch.ops.aten.view.default(addmm_17, [1, 8, 768]);  addmm_17 = None
        add_22 = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_18 = var_mean_6[0]
        getitem_19 = var_mean_6[1];  var_mean_6 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7 = torch.ops.aten.sub.Tensor(add_22, getitem_19);  add_22 = getitem_19 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
        mul_22 = torch.ops.aten.mul.Tensor(mul_21, primals_56)
        add_24 = torch.ops.aten.add.Tensor(mul_22, primals_57);  mul_22 = primals_57 = None
        view_48 = torch.ops.aten.view.default(add_24, [8, 768])
        permute_30 = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_18 = torch.ops.aten.addmm.default(primals_59, view_48, permute_30);  primals_59 = permute_30 = None
        view_49 = torch.ops.aten.view.default(addmm_18, [1, 8, 768]);  addmm_18 = None
        view_50 = torch.ops.aten.view.default(view_49, [1, -1, 12, 64]);  view_49 = None
        permute_31 = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_32 = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_19 = torch.ops.aten.addmm.default(primals_61, view_48, permute_32);  primals_61 = permute_32 = None
        view_52 = torch.ops.aten.view.default(addmm_19, [1, 8, 768]);  addmm_19 = None
        view_53 = torch.ops.aten.view.default(view_52, [1, -1, 12, 64]);  view_52 = None
        permute_33 = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        permute_34 = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_20 = torch.ops.aten.addmm.default(primals_63, view_48, permute_34);  primals_63 = permute_34 = None
        view_55 = torch.ops.aten.view.default(addmm_20, [1, 8, 768]);  addmm_20 = None
        view_56 = torch.ops.aten.view.default(view_55, [1, -1, 12, 64]);  view_55 = None
        permute_35 = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        _scaled_dot_product_flash_attention_for_cpu_3 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_31, permute_33, permute_35, attn_mask = where)
        getitem_20 = _scaled_dot_product_flash_attention_for_cpu_3[0]
        getitem_21 = _scaled_dot_product_flash_attention_for_cpu_3[1];  _scaled_dot_product_flash_attention_for_cpu_3 = None
        permute_36 = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3])
        view_57 = torch.ops.aten.view.default(permute_36, [1, 8, 768]);  permute_36 = None
        view_58 = torch.ops.aten.view.default(view_57, [8, 768]);  view_57 = None
        permute_37 = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_21 = torch.ops.aten.addmm.default(primals_65, view_58, permute_37);  primals_65 = view_58 = permute_37 = None
        view_59 = torch.ops.aten.view.default(addmm_21, [1, 8, 768]);  addmm_21 = None
        add_25 = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_22 = var_mean_7[0]
        getitem_23 = var_mean_7[1];  var_mean_7 = None
        add_26 = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_25, getitem_23);  add_25 = getitem_23 = None
        mul_23 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_23, primals_66)
        add_27 = torch.ops.aten.add.Tensor(mul_24, primals_67);  mul_24 = primals_67 = None
        view_60 = torch.ops.aten.view.default(add_27, [8, 768])
        permute_38 = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_22 = torch.ops.aten.addmm.default(primals_69, view_60, permute_38);  primals_69 = permute_38 = None
        view_61 = torch.ops.aten.view.default(addmm_22, [1, 8, 3072])
        mul_25 = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26 = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3 = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28 = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        view_62 = torch.ops.aten.view.default(mul_27, [8, 3072]);  mul_27 = None
        permute_39 = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_23 = torch.ops.aten.addmm.default(primals_71, view_62, permute_39);  primals_71 = permute_39 = None
        view_63 = torch.ops.aten.view.default(addmm_23, [1, 8, 768]);  addmm_23 = None
        add_29 = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_24 = var_mean_8[0]
        getitem_25 = var_mean_8[1];  var_mean_8 = None
        add_30 = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9 = torch.ops.aten.sub.Tensor(add_29, getitem_25);  add_29 = getitem_25 = None
        mul_28 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
        mul_29 = torch.ops.aten.mul.Tensor(mul_28, primals_72)
        add_31 = torch.ops.aten.add.Tensor(mul_29, primals_73);  mul_29 = primals_73 = None
        view_64 = torch.ops.aten.view.default(add_31, [8, 768])
        permute_40 = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_24 = torch.ops.aten.addmm.default(primals_75, view_64, permute_40);  primals_75 = permute_40 = None
        view_65 = torch.ops.aten.view.default(addmm_24, [1, 8, 768]);  addmm_24 = None
        view_66 = torch.ops.aten.view.default(view_65, [1, -1, 12, 64]);  view_65 = None
        permute_41 = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        permute_42 = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_25 = torch.ops.aten.addmm.default(primals_77, view_64, permute_42);  primals_77 = permute_42 = None
        view_68 = torch.ops.aten.view.default(addmm_25, [1, 8, 768]);  addmm_25 = None
        view_69 = torch.ops.aten.view.default(view_68, [1, -1, 12, 64]);  view_68 = None
        permute_43 = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        permute_44 = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_26 = torch.ops.aten.addmm.default(primals_79, view_64, permute_44);  primals_79 = permute_44 = None
        view_71 = torch.ops.aten.view.default(addmm_26, [1, 8, 768]);  addmm_26 = None
        view_72 = torch.ops.aten.view.default(view_71, [1, -1, 12, 64]);  view_71 = None
        permute_45 = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        _scaled_dot_product_flash_attention_for_cpu_4 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_41, permute_43, permute_45, attn_mask = where)
        getitem_26 = _scaled_dot_product_flash_attention_for_cpu_4[0]
        getitem_27 = _scaled_dot_product_flash_attention_for_cpu_4[1];  _scaled_dot_product_flash_attention_for_cpu_4 = None
        permute_46 = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3])
        view_73 = torch.ops.aten.view.default(permute_46, [1, 8, 768]);  permute_46 = None
        view_74 = torch.ops.aten.view.default(view_73, [8, 768]);  view_73 = None
        permute_47 = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_27 = torch.ops.aten.addmm.default(primals_81, view_74, permute_47);  primals_81 = view_74 = permute_47 = None
        view_75 = torch.ops.aten.view.default(addmm_27, [1, 8, 768]);  addmm_27 = None
        add_32 = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_28 = var_mean_9[0]
        getitem_29 = var_mean_9[1];  var_mean_9 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_32, getitem_29);  add_32 = getitem_29 = None
        mul_30 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, primals_82)
        add_34 = torch.ops.aten.add.Tensor(mul_31, primals_83);  mul_31 = primals_83 = None
        view_76 = torch.ops.aten.view.default(add_34, [8, 768])
        permute_48 = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_28 = torch.ops.aten.addmm.default(primals_85, view_76, permute_48);  primals_85 = permute_48 = None
        view_77 = torch.ops.aten.view.default(addmm_28, [1, 8, 3072])
        mul_32 = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33 = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4 = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35 = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        view_78 = torch.ops.aten.view.default(mul_34, [8, 3072]);  mul_34 = None
        permute_49 = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_29 = torch.ops.aten.addmm.default(primals_87, view_78, permute_49);  primals_87 = permute_49 = None
        view_79 = torch.ops.aten.view.default(addmm_29, [1, 8, 768]);  addmm_29 = None
        add_36 = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_30 = var_mean_10[0]
        getitem_31 = var_mean_10[1];  var_mean_10 = None
        add_37 = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_36, getitem_31);  add_36 = getitem_31 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, primals_88)
        add_38 = torch.ops.aten.add.Tensor(mul_36, primals_89);  mul_36 = primals_89 = None
        view_80 = torch.ops.aten.view.default(add_38, [8, 768])
        permute_50 = torch.ops.aten.permute.default(primals_90, [1, 0])
        addmm_30 = torch.ops.aten.addmm.default(primals_91, view_80, permute_50);  primals_91 = permute_50 = None
        view_81 = torch.ops.aten.view.default(addmm_30, [1, 8, 768]);  addmm_30 = None
        view_82 = torch.ops.aten.view.default(view_81, [1, -1, 12, 64]);  view_81 = None
        permute_51 = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        permute_52 = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_31 = torch.ops.aten.addmm.default(primals_93, view_80, permute_52);  primals_93 = permute_52 = None
        view_84 = torch.ops.aten.view.default(addmm_31, [1, 8, 768]);  addmm_31 = None
        view_85 = torch.ops.aten.view.default(view_84, [1, -1, 12, 64]);  view_84 = None
        permute_53 = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        permute_54 = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_32 = torch.ops.aten.addmm.default(primals_95, view_80, permute_54);  primals_95 = permute_54 = None
        view_87 = torch.ops.aten.view.default(addmm_32, [1, 8, 768]);  addmm_32 = None
        view_88 = torch.ops.aten.view.default(view_87, [1, -1, 12, 64]);  view_87 = None
        permute_55 = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        _scaled_dot_product_flash_attention_for_cpu_5 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_51, permute_53, permute_55, attn_mask = where)
        getitem_32 = _scaled_dot_product_flash_attention_for_cpu_5[0]
        getitem_33 = _scaled_dot_product_flash_attention_for_cpu_5[1];  _scaled_dot_product_flash_attention_for_cpu_5 = None
        permute_56 = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3])
        view_89 = torch.ops.aten.view.default(permute_56, [1, 8, 768]);  permute_56 = None
        view_90 = torch.ops.aten.view.default(view_89, [8, 768]);  view_89 = None
        permute_57 = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_33 = torch.ops.aten.addmm.default(primals_97, view_90, permute_57);  primals_97 = view_90 = permute_57 = None
        view_91 = torch.ops.aten.view.default(addmm_33, [1, 8, 768]);  addmm_33 = None
        add_39 = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_34 = var_mean_11[0]
        getitem_35 = var_mean_11[1];  var_mean_11 = None
        add_40 = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12 = torch.ops.aten.sub.Tensor(add_39, getitem_35);  add_39 = getitem_35 = None
        mul_37 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
        mul_38 = torch.ops.aten.mul.Tensor(mul_37, primals_98)
        add_41 = torch.ops.aten.add.Tensor(mul_38, primals_99);  mul_38 = primals_99 = None
        view_92 = torch.ops.aten.view.default(add_41, [8, 768])
        permute_58 = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_34 = torch.ops.aten.addmm.default(primals_101, view_92, permute_58);  primals_101 = permute_58 = None
        view_93 = torch.ops.aten.view.default(addmm_34, [1, 8, 3072])
        mul_39 = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40 = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5 = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42 = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        view_94 = torch.ops.aten.view.default(mul_41, [8, 3072]);  mul_41 = None
        permute_59 = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm_35 = torch.ops.aten.addmm.default(primals_103, view_94, permute_59);  primals_103 = permute_59 = None
        view_95 = torch.ops.aten.view.default(addmm_35, [1, 8, 768]);  addmm_35 = None
        add_43 = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_36 = var_mean_12[0]
        getitem_37 = var_mean_12[1];  var_mean_12 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_13 = torch.ops.aten.sub.Tensor(add_43, getitem_37);  add_43 = getitem_37 = None
        mul_42 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_42, primals_104)
        add_45 = torch.ops.aten.add.Tensor(mul_43, primals_105);  mul_43 = primals_105 = None
        view_96 = torch.ops.aten.view.default(add_45, [8, 768])
        permute_60 = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_36 = torch.ops.aten.addmm.default(primals_107, view_96, permute_60);  primals_107 = permute_60 = None
        view_97 = torch.ops.aten.view.default(addmm_36, [1, 8, 768]);  addmm_36 = None
        view_98 = torch.ops.aten.view.default(view_97, [1, -1, 12, 64]);  view_97 = None
        permute_61 = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        permute_62 = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_37 = torch.ops.aten.addmm.default(primals_109, view_96, permute_62);  primals_109 = permute_62 = None
        view_100 = torch.ops.aten.view.default(addmm_37, [1, 8, 768]);  addmm_37 = None
        view_101 = torch.ops.aten.view.default(view_100, [1, -1, 12, 64]);  view_100 = None
        permute_63 = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        permute_64 = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_38 = torch.ops.aten.addmm.default(primals_111, view_96, permute_64);  primals_111 = permute_64 = None
        view_103 = torch.ops.aten.view.default(addmm_38, [1, 8, 768]);  addmm_38 = None
        view_104 = torch.ops.aten.view.default(view_103, [1, -1, 12, 64]);  view_103 = None
        permute_65 = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        _scaled_dot_product_flash_attention_for_cpu_6 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_61, permute_63, permute_65, attn_mask = where)
        getitem_38 = _scaled_dot_product_flash_attention_for_cpu_6[0]
        getitem_39 = _scaled_dot_product_flash_attention_for_cpu_6[1];  _scaled_dot_product_flash_attention_for_cpu_6 = None
        permute_66 = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        view_105 = torch.ops.aten.view.default(permute_66, [1, 8, 768]);  permute_66 = None
        view_106 = torch.ops.aten.view.default(view_105, [8, 768]);  view_105 = None
        permute_67 = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_39 = torch.ops.aten.addmm.default(primals_113, view_106, permute_67);  primals_113 = view_106 = permute_67 = None
        view_107 = torch.ops.aten.view.default(addmm_39, [1, 8, 768]);  addmm_39 = None
        add_46 = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_40 = var_mean_13[0]
        getitem_41 = var_mean_13[1];  var_mean_13 = None
        add_47 = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_46, getitem_41);  add_46 = getitem_41 = None
        mul_44 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, primals_114)
        add_48 = torch.ops.aten.add.Tensor(mul_45, primals_115);  mul_45 = primals_115 = None
        view_108 = torch.ops.aten.view.default(add_48, [8, 768])
        permute_68 = torch.ops.aten.permute.default(primals_116, [1, 0])
        addmm_40 = torch.ops.aten.addmm.default(primals_117, view_108, permute_68);  primals_117 = permute_68 = None
        view_109 = torch.ops.aten.view.default(addmm_40, [1, 8, 3072])
        mul_46 = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47 = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6 = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49 = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        view_110 = torch.ops.aten.view.default(mul_48, [8, 3072]);  mul_48 = None
        permute_69 = torch.ops.aten.permute.default(primals_118, [1, 0])
        addmm_41 = torch.ops.aten.addmm.default(primals_119, view_110, permute_69);  primals_119 = permute_69 = None
        view_111 = torch.ops.aten.view.default(addmm_41, [1, 8, 768]);  addmm_41 = None
        add_50 = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_42 = var_mean_14[0]
        getitem_43 = var_mean_14[1];  var_mean_14 = None
        add_51 = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_15 = torch.ops.aten.sub.Tensor(add_50, getitem_43);  add_50 = getitem_43 = None
        mul_49 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = None
        mul_50 = torch.ops.aten.mul.Tensor(mul_49, primals_120)
        add_52 = torch.ops.aten.add.Tensor(mul_50, primals_121);  mul_50 = primals_121 = None
        view_112 = torch.ops.aten.view.default(add_52, [8, 768])
        permute_70 = torch.ops.aten.permute.default(primals_122, [1, 0])
        addmm_42 = torch.ops.aten.addmm.default(primals_123, view_112, permute_70);  primals_123 = permute_70 = None
        view_113 = torch.ops.aten.view.default(addmm_42, [1, 8, 768]);  addmm_42 = None
        view_114 = torch.ops.aten.view.default(view_113, [1, -1, 12, 64]);  view_113 = None
        permute_71 = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        permute_72 = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_43 = torch.ops.aten.addmm.default(primals_125, view_112, permute_72);  primals_125 = permute_72 = None
        view_116 = torch.ops.aten.view.default(addmm_43, [1, 8, 768]);  addmm_43 = None
        view_117 = torch.ops.aten.view.default(view_116, [1, -1, 12, 64]);  view_116 = None
        permute_73 = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        permute_74 = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_44 = torch.ops.aten.addmm.default(primals_127, view_112, permute_74);  primals_127 = permute_74 = None
        view_119 = torch.ops.aten.view.default(addmm_44, [1, 8, 768]);  addmm_44 = None
        view_120 = torch.ops.aten.view.default(view_119, [1, -1, 12, 64]);  view_119 = None
        permute_75 = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        _scaled_dot_product_flash_attention_for_cpu_7 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_71, permute_73, permute_75, attn_mask = where)
        getitem_44 = _scaled_dot_product_flash_attention_for_cpu_7[0]
        getitem_45 = _scaled_dot_product_flash_attention_for_cpu_7[1];  _scaled_dot_product_flash_attention_for_cpu_7 = None
        permute_76 = torch.ops.aten.permute.default(getitem_44, [0, 2, 1, 3])
        view_121 = torch.ops.aten.view.default(permute_76, [1, 8, 768]);  permute_76 = None
        view_122 = torch.ops.aten.view.default(view_121, [8, 768]);  view_121 = None
        permute_77 = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_45 = torch.ops.aten.addmm.default(primals_129, view_122, permute_77);  primals_129 = view_122 = permute_77 = None
        view_123 = torch.ops.aten.view.default(addmm_45, [1, 8, 768]);  addmm_45 = None
        add_53 = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_46 = var_mean_15[0]
        getitem_47 = var_mean_15[1];  var_mean_15 = None
        add_54 = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_16 = torch.ops.aten.sub.Tensor(add_53, getitem_47);  add_53 = getitem_47 = None
        mul_51 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = None
        mul_52 = torch.ops.aten.mul.Tensor(mul_51, primals_130)
        add_55 = torch.ops.aten.add.Tensor(mul_52, primals_131);  mul_52 = primals_131 = None
        view_124 = torch.ops.aten.view.default(add_55, [8, 768])
        permute_78 = torch.ops.aten.permute.default(primals_132, [1, 0])
        addmm_46 = torch.ops.aten.addmm.default(primals_133, view_124, permute_78);  primals_133 = permute_78 = None
        view_125 = torch.ops.aten.view.default(addmm_46, [1, 8, 3072])
        mul_53 = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54 = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7 = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56 = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        view_126 = torch.ops.aten.view.default(mul_55, [8, 3072]);  mul_55 = None
        permute_79 = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_47 = torch.ops.aten.addmm.default(primals_135, view_126, permute_79);  primals_135 = permute_79 = None
        view_127 = torch.ops.aten.view.default(addmm_47, [1, 8, 768]);  addmm_47 = None
        add_57 = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_48 = var_mean_16[0]
        getitem_49 = var_mean_16[1];  var_mean_16 = None
        add_58 = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_17 = torch.ops.aten.sub.Tensor(add_57, getitem_49);  add_57 = getitem_49 = None
        mul_56 = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = None
        mul_57 = torch.ops.aten.mul.Tensor(mul_56, primals_136)
        add_59 = torch.ops.aten.add.Tensor(mul_57, primals_137);  mul_57 = primals_137 = None
        view_128 = torch.ops.aten.view.default(add_59, [8, 768])
        permute_80 = torch.ops.aten.permute.default(primals_138, [1, 0])
        addmm_48 = torch.ops.aten.addmm.default(primals_139, view_128, permute_80);  primals_139 = permute_80 = None
        view_129 = torch.ops.aten.view.default(addmm_48, [1, 8, 768]);  addmm_48 = None
        view_130 = torch.ops.aten.view.default(view_129, [1, -1, 12, 64]);  view_129 = None
        permute_81 = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        permute_82 = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_49 = torch.ops.aten.addmm.default(primals_141, view_128, permute_82);  primals_141 = permute_82 = None
        view_132 = torch.ops.aten.view.default(addmm_49, [1, 8, 768]);  addmm_49 = None
        view_133 = torch.ops.aten.view.default(view_132, [1, -1, 12, 64]);  view_132 = None
        permute_83 = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        permute_84 = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_50 = torch.ops.aten.addmm.default(primals_143, view_128, permute_84);  primals_143 = permute_84 = None
        view_135 = torch.ops.aten.view.default(addmm_50, [1, 8, 768]);  addmm_50 = None
        view_136 = torch.ops.aten.view.default(view_135, [1, -1, 12, 64]);  view_135 = None
        permute_85 = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        _scaled_dot_product_flash_attention_for_cpu_8 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_81, permute_83, permute_85, attn_mask = where)
        getitem_50 = _scaled_dot_product_flash_attention_for_cpu_8[0]
        getitem_51 = _scaled_dot_product_flash_attention_for_cpu_8[1];  _scaled_dot_product_flash_attention_for_cpu_8 = None
        permute_86 = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3])
        view_137 = torch.ops.aten.view.default(permute_86, [1, 8, 768]);  permute_86 = None
        view_138 = torch.ops.aten.view.default(view_137, [8, 768]);  view_137 = None
        permute_87 = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_51 = torch.ops.aten.addmm.default(primals_145, view_138, permute_87);  primals_145 = view_138 = permute_87 = None
        view_139 = torch.ops.aten.view.default(addmm_51, [1, 8, 768]);  addmm_51 = None
        add_60 = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_52 = var_mean_17[0]
        getitem_53 = var_mean_17[1];  var_mean_17 = None
        add_61 = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_17 = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_18 = torch.ops.aten.sub.Tensor(add_60, getitem_53);  add_60 = getitem_53 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, primals_146)
        add_62 = torch.ops.aten.add.Tensor(mul_59, primals_147);  mul_59 = primals_147 = None
        view_140 = torch.ops.aten.view.default(add_62, [8, 768])
        permute_88 = torch.ops.aten.permute.default(primals_148, [1, 0])
        addmm_52 = torch.ops.aten.addmm.default(primals_149, view_140, permute_88);  primals_149 = permute_88 = None
        view_141 = torch.ops.aten.view.default(addmm_52, [1, 8, 3072])
        mul_60 = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61 = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8 = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63 = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        view_142 = torch.ops.aten.view.default(mul_62, [8, 3072]);  mul_62 = None
        permute_89 = torch.ops.aten.permute.default(primals_150, [1, 0])
        addmm_53 = torch.ops.aten.addmm.default(primals_151, view_142, permute_89);  primals_151 = permute_89 = None
        view_143 = torch.ops.aten.view.default(addmm_53, [1, 8, 768]);  addmm_53 = None
        add_64 = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_54 = var_mean_18[0]
        getitem_55 = var_mean_18[1];  var_mean_18 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_18 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_19 = torch.ops.aten.sub.Tensor(add_64, getitem_55);  add_64 = getitem_55 = None
        mul_63 = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_63, primals_152)
        add_66 = torch.ops.aten.add.Tensor(mul_64, primals_153);  mul_64 = primals_153 = None
        view_144 = torch.ops.aten.view.default(add_66, [8, 768])
        permute_90 = torch.ops.aten.permute.default(primals_154, [1, 0])
        addmm_54 = torch.ops.aten.addmm.default(primals_155, view_144, permute_90);  primals_155 = permute_90 = None
        view_145 = torch.ops.aten.view.default(addmm_54, [1, 8, 768]);  addmm_54 = None
        view_146 = torch.ops.aten.view.default(view_145, [1, -1, 12, 64]);  view_145 = None
        permute_91 = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        permute_92 = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_55 = torch.ops.aten.addmm.default(primals_157, view_144, permute_92);  primals_157 = permute_92 = None
        view_148 = torch.ops.aten.view.default(addmm_55, [1, 8, 768]);  addmm_55 = None
        view_149 = torch.ops.aten.view.default(view_148, [1, -1, 12, 64]);  view_148 = None
        permute_93 = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        permute_94 = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_56 = torch.ops.aten.addmm.default(primals_159, view_144, permute_94);  primals_159 = permute_94 = None
        view_151 = torch.ops.aten.view.default(addmm_56, [1, 8, 768]);  addmm_56 = None
        view_152 = torch.ops.aten.view.default(view_151, [1, -1, 12, 64]);  view_151 = None
        permute_95 = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        _scaled_dot_product_flash_attention_for_cpu_9 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_91, permute_93, permute_95, attn_mask = where)
        getitem_56 = _scaled_dot_product_flash_attention_for_cpu_9[0]
        getitem_57 = _scaled_dot_product_flash_attention_for_cpu_9[1];  _scaled_dot_product_flash_attention_for_cpu_9 = None
        permute_96 = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3])
        view_153 = torch.ops.aten.view.default(permute_96, [1, 8, 768]);  permute_96 = None
        view_154 = torch.ops.aten.view.default(view_153, [8, 768]);  view_153 = None
        permute_97 = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_57 = torch.ops.aten.addmm.default(primals_161, view_154, permute_97);  primals_161 = view_154 = permute_97 = None
        view_155 = torch.ops.aten.view.default(addmm_57, [1, 8, 768]);  addmm_57 = None
        add_67 = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_58 = var_mean_19[0]
        getitem_59 = var_mean_19[1];  var_mean_19 = None
        add_68 = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_19 = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_20 = torch.ops.aten.sub.Tensor(add_67, getitem_59);  add_67 = getitem_59 = None
        mul_65 = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_65, primals_162)
        add_69 = torch.ops.aten.add.Tensor(mul_66, primals_163);  mul_66 = primals_163 = None
        view_156 = torch.ops.aten.view.default(add_69, [8, 768])
        permute_98 = torch.ops.aten.permute.default(primals_164, [1, 0])
        addmm_58 = torch.ops.aten.addmm.default(primals_165, view_156, permute_98);  primals_165 = permute_98 = None
        view_157 = torch.ops.aten.view.default(addmm_58, [1, 8, 3072])
        mul_67 = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68 = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9 = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70 = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        view_158 = torch.ops.aten.view.default(mul_69, [8, 3072]);  mul_69 = None
        permute_99 = torch.ops.aten.permute.default(primals_166, [1, 0])
        addmm_59 = torch.ops.aten.addmm.default(primals_167, view_158, permute_99);  primals_167 = permute_99 = None
        view_159 = torch.ops.aten.view.default(addmm_59, [1, 8, 768]);  addmm_59 = None
        add_71 = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_60 = var_mean_20[0]
        getitem_61 = var_mean_20[1];  var_mean_20 = None
        add_72 = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_20 = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_21 = torch.ops.aten.sub.Tensor(add_71, getitem_61);  add_71 = getitem_61 = None
        mul_70 = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = None
        mul_71 = torch.ops.aten.mul.Tensor(mul_70, primals_168)
        add_73 = torch.ops.aten.add.Tensor(mul_71, primals_169);  mul_71 = primals_169 = None
        view_160 = torch.ops.aten.view.default(add_73, [8, 768])
        permute_100 = torch.ops.aten.permute.default(primals_170, [1, 0])
        addmm_60 = torch.ops.aten.addmm.default(primals_171, view_160, permute_100);  primals_171 = permute_100 = None
        view_161 = torch.ops.aten.view.default(addmm_60, [1, 8, 768]);  addmm_60 = None
        view_162 = torch.ops.aten.view.default(view_161, [1, -1, 12, 64]);  view_161 = None
        permute_101 = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        permute_102 = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_61 = torch.ops.aten.addmm.default(primals_173, view_160, permute_102);  primals_173 = permute_102 = None
        view_164 = torch.ops.aten.view.default(addmm_61, [1, 8, 768]);  addmm_61 = None
        view_165 = torch.ops.aten.view.default(view_164, [1, -1, 12, 64]);  view_164 = None
        permute_103 = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        permute_104 = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_62 = torch.ops.aten.addmm.default(primals_175, view_160, permute_104);  primals_175 = permute_104 = None
        view_167 = torch.ops.aten.view.default(addmm_62, [1, 8, 768]);  addmm_62 = None
        view_168 = torch.ops.aten.view.default(view_167, [1, -1, 12, 64]);  view_167 = None
        permute_105 = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        _scaled_dot_product_flash_attention_for_cpu_10 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_101, permute_103, permute_105, attn_mask = where)
        getitem_62 = _scaled_dot_product_flash_attention_for_cpu_10[0]
        getitem_63 = _scaled_dot_product_flash_attention_for_cpu_10[1];  _scaled_dot_product_flash_attention_for_cpu_10 = None
        permute_106 = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3])
        view_169 = torch.ops.aten.view.default(permute_106, [1, 8, 768]);  permute_106 = None
        view_170 = torch.ops.aten.view.default(view_169, [8, 768]);  view_169 = None
        permute_107 = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_63 = torch.ops.aten.addmm.default(primals_177, view_170, permute_107);  primals_177 = view_170 = permute_107 = None
        view_171 = torch.ops.aten.view.default(addmm_63, [1, 8, 768]);  addmm_63 = None
        add_74 = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_64 = var_mean_21[0]
        getitem_65 = var_mean_21[1];  var_mean_21 = None
        add_75 = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_21 = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_22 = torch.ops.aten.sub.Tensor(add_74, getitem_65);  add_74 = getitem_65 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = None
        mul_73 = torch.ops.aten.mul.Tensor(mul_72, primals_178)
        add_76 = torch.ops.aten.add.Tensor(mul_73, primals_179);  mul_73 = primals_179 = None
        view_172 = torch.ops.aten.view.default(add_76, [8, 768])
        permute_108 = torch.ops.aten.permute.default(primals_180, [1, 0])
        addmm_64 = torch.ops.aten.addmm.default(primals_181, view_172, permute_108);  primals_181 = permute_108 = None
        view_173 = torch.ops.aten.view.default(addmm_64, [1, 8, 3072])
        mul_74 = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75 = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10 = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77 = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        view_174 = torch.ops.aten.view.default(mul_76, [8, 3072]);  mul_76 = None
        permute_109 = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_65 = torch.ops.aten.addmm.default(primals_183, view_174, permute_109);  primals_183 = permute_109 = None
        view_175 = torch.ops.aten.view.default(addmm_65, [1, 8, 768]);  addmm_65 = None
        add_78 = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_66 = var_mean_22[0]
        getitem_67 = var_mean_22[1];  var_mean_22 = None
        add_79 = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_22 = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_23 = torch.ops.aten.sub.Tensor(add_78, getitem_67);  add_78 = getitem_67 = None
        mul_77 = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
        mul_78 = torch.ops.aten.mul.Tensor(mul_77, primals_184)
        add_80 = torch.ops.aten.add.Tensor(mul_78, primals_185);  mul_78 = primals_185 = None
        view_176 = torch.ops.aten.view.default(add_80, [8, 768])
        permute_110 = torch.ops.aten.permute.default(primals_186, [1, 0])
        addmm_66 = torch.ops.aten.addmm.default(primals_187, view_176, permute_110);  primals_187 = permute_110 = None
        view_177 = torch.ops.aten.view.default(addmm_66, [1, 8, 768]);  addmm_66 = None
        view_178 = torch.ops.aten.view.default(view_177, [1, -1, 12, 64]);  view_177 = None
        permute_111 = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_112 = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_67 = torch.ops.aten.addmm.default(primals_189, view_176, permute_112);  primals_189 = permute_112 = None
        view_180 = torch.ops.aten.view.default(addmm_67, [1, 8, 768]);  addmm_67 = None
        view_181 = torch.ops.aten.view.default(view_180, [1, -1, 12, 64]);  view_180 = None
        permute_113 = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        permute_114 = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_68 = torch.ops.aten.addmm.default(primals_191, view_176, permute_114);  primals_191 = permute_114 = None
        view_183 = torch.ops.aten.view.default(addmm_68, [1, 8, 768]);  addmm_68 = None
        view_184 = torch.ops.aten.view.default(view_183, [1, -1, 12, 64]);  view_183 = None
        permute_115 = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_flash_attention_for_cpu_11 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_111, permute_113, permute_115, attn_mask = where)
        getitem_68 = _scaled_dot_product_flash_attention_for_cpu_11[0]
        getitem_69 = _scaled_dot_product_flash_attention_for_cpu_11[1];  _scaled_dot_product_flash_attention_for_cpu_11 = None
        permute_116 = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3])
        view_185 = torch.ops.aten.view.default(permute_116, [1, 8, 768]);  permute_116 = None
        view_186 = torch.ops.aten.view.default(view_185, [8, 768]);  view_185 = None
        permute_117 = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_69 = torch.ops.aten.addmm.default(primals_193, view_186, permute_117);  primals_193 = view_186 = permute_117 = None
        view_187 = torch.ops.aten.view.default(addmm_69, [1, 8, 768]);  addmm_69 = None
        add_81 = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_70 = var_mean_23[0]
        getitem_71 = var_mean_23[1];  var_mean_23 = None
        add_82 = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_23 = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_24 = torch.ops.aten.sub.Tensor(add_81, getitem_71);  add_81 = getitem_71 = None
        mul_79 = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
        mul_80 = torch.ops.aten.mul.Tensor(mul_79, primals_194)
        add_83 = torch.ops.aten.add.Tensor(mul_80, primals_195);  mul_80 = primals_195 = None
        view_188 = torch.ops.aten.view.default(add_83, [8, 768])
        permute_118 = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_70 = torch.ops.aten.addmm.default(primals_197, view_188, permute_118);  primals_197 = permute_118 = None
        view_189 = torch.ops.aten.view.default(addmm_70, [1, 8, 3072])
        mul_81 = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82 = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11 = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84 = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        view_190 = torch.ops.aten.view.default(mul_83, [8, 3072]);  mul_83 = None
        permute_119 = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_71 = torch.ops.aten.addmm.default(primals_199, view_190, permute_119);  primals_199 = permute_119 = None
        view_191 = torch.ops.aten.view.default(addmm_71, [1, 8, 768]);  addmm_71 = None
        add_85 = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_72 = var_mean_24[0]
        getitem_73 = var_mean_24[1];  var_mean_24 = None
        add_86 = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_24 = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_25 = torch.ops.aten.sub.Tensor(add_85, getitem_73);  add_85 = getitem_73 = None
        mul_84 = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_84, primals_200)
        add_87 = torch.ops.aten.add.Tensor(mul_85, primals_201);  mul_85 = primals_201 = None
        select = torch.ops.aten.select.int(add_87, 1, 0)
        permute_120 = torch.ops.aten.permute.default(primals_202, [1, 0])
        addmm_72 = torch.ops.aten.addmm.default(primals_203, select, permute_120);  primals_203 = permute_120 = None
        tanh = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None
        div = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        div_1 = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        div_2 = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        div_3 = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        div_4 = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        div_5 = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        div_6 = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        div_7 = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        div_8 = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        div_9 = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        div_10 = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        div_11 = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        div_12 = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        div_13 = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        div_14 = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        div_15 = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        div_16 = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        div_17 = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        div_18 = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        div_19 = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        div_20 = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        div_21 = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        div_22 = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        div_23 = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        div_24 = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (add_87, tanh, primals_1, primals_2, primals_3, primals_7, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, mul, where, view, permute_1, permute_3, permute_5, getitem_2, getitem_3, mul_2, view_12, addmm_4, view_14, mul_7, view_16, permute_11, permute_13, permute_15, getitem_8, getitem_9, mul_9, view_28, addmm_10, view_30, mul_14, view_32, permute_21, permute_23, permute_25, getitem_14, getitem_15, mul_16, view_44, addmm_16, view_46, mul_21, view_48, permute_31, permute_33, permute_35, getitem_20, getitem_21, mul_23, view_60, addmm_22, view_62, mul_28, view_64, permute_41, permute_43, permute_45, getitem_26, getitem_27, mul_30, view_76, addmm_28, view_78, mul_35, view_80, permute_51, permute_53, permute_55, getitem_32, getitem_33, mul_37, view_92, addmm_34, view_94, mul_42, view_96, permute_61, permute_63, permute_65, getitem_38, getitem_39, mul_44, view_108, addmm_40, view_110, mul_49, view_112, permute_71, permute_73, permute_75, getitem_44, getitem_45, mul_51, view_124, addmm_46, view_126, mul_56, view_128, permute_81, permute_83, permute_85, getitem_50, getitem_51, mul_58, view_140, addmm_52, view_142, mul_63, view_144, permute_91, permute_93, permute_95, getitem_56, getitem_57, mul_65, view_156, addmm_58, view_158, mul_70, view_160, permute_101, permute_103, permute_105, getitem_62, getitem_63, mul_72, view_172, addmm_64, view_174, mul_77, view_176, permute_111, permute_113, permute_115, getitem_68, getitem_69, mul_79, view_188, addmm_70, view_190, mul_84, select, tanh, div, div_1, div_2, div_3, div_4, div_5, div_6, div_7, div_8, div_9, div_10, div_11, div_12, div_13, div_14, div_15, div_16, div_17, div_18, div_19, div_20, div_21, div_22, div_23, div_24)
        
def load_args(reader):
    buf0 = reader.storage(None, 64, dtype_hint=torch.int64)
    reader.tensor(buf0, (1, 8), dtype=torch.int64, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf1, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_2
    buf2 = reader.storage(None, 4096, dtype_hint=torch.int64)
    reader.tensor(buf2, (1, 512), dtype=torch.int64, is_leaf=True)  # primals_3
    buf3 = reader.storage(None, 93763584)
    reader.tensor(buf3, (30522, 768), is_leaf=True)  # primals_4
    buf4 = reader.storage(None, 6144)
    reader.tensor(buf4, (2, 768), is_leaf=True)  # primals_5
    buf5 = reader.storage(None, 1572864)
    reader.tensor(buf5, (512, 768), is_leaf=True)  # primals_6
    buf6 = reader.storage(None, 3072)
    reader.tensor(buf6, (768,), is_leaf=True)  # primals_7
    buf7 = reader.storage(None, 3072)
    reader.tensor(buf7, (768,), is_leaf=True)  # primals_8
    buf8 = reader.storage(None, 64, dtype_hint=torch.int64)
    reader.tensor(buf8, (1, 8), dtype=torch.int64, is_leaf=True)  # primals_9
    buf9 = reader.storage(None, 2359296)
    reader.tensor(buf9, (768, 768), is_leaf=True)  # primals_10
    buf10 = reader.storage(None, 3072)
    reader.tensor(buf10, (768,), is_leaf=True)  # primals_11
    buf11 = reader.storage(None, 2359296)
    reader.tensor(buf11, (768, 768), is_leaf=True)  # primals_12
    buf12 = reader.storage(None, 3072)
    reader.tensor(buf12, (768,), is_leaf=True)  # primals_13
    buf13 = reader.storage(None, 2359296)
    reader.tensor(buf13, (768, 768), is_leaf=True)  # primals_14
    buf14 = reader.storage(None, 3072)
    reader.tensor(buf14, (768,), is_leaf=True)  # primals_15
    buf15 = reader.storage(None, 2359296)
    reader.tensor(buf15, (768, 768), is_leaf=True)  # primals_16
    buf16 = reader.storage(None, 3072)
    reader.tensor(buf16, (768,), is_leaf=True)  # primals_17
    buf17 = reader.storage(None, 3072)
    reader.tensor(buf17, (768,), is_leaf=True)  # primals_18
    buf18 = reader.storage(None, 3072)
    reader.tensor(buf18, (768,), is_leaf=True)  # primals_19
    buf19 = reader.storage(None, 9437184)
    reader.tensor(buf19, (3072, 768), is_leaf=True)  # primals_20
    buf20 = reader.storage(None, 12288)
    reader.tensor(buf20, (3072,), is_leaf=True)  # primals_21
    buf21 = reader.storage(None, 9437184)
    reader.tensor(buf21, (768, 3072), is_leaf=True)  # primals_22
    buf22 = reader.storage(None, 3072)
    reader.tensor(buf22, (768,), is_leaf=True)  # primals_23
    buf23 = reader.storage(None, 3072)
    reader.tensor(buf23, (768,), is_leaf=True)  # primals_24
    buf24 = reader.storage(None, 3072)
    reader.tensor(buf24, (768,), is_leaf=True)  # primals_25
    buf25 = reader.storage(None, 2359296)
    reader.tensor(buf25, (768, 768), is_leaf=True)  # primals_26
    buf26 = reader.storage(None, 3072)
    reader.tensor(buf26, (768,), is_leaf=True)  # primals_27
    buf27 = reader.storage(None, 2359296)
    reader.tensor(buf27, (768, 768), is_leaf=True)  # primals_28
    buf28 = reader.storage(None, 3072)
    reader.tensor(buf28, (768,), is_leaf=True)  # primals_29
    buf29 = reader.storage(None, 2359296)
    reader.tensor(buf29, (768, 768), is_leaf=True)  # primals_30
    buf30 = reader.storage(None, 3072)
    reader.tensor(buf30, (768,), is_leaf=True)  # primals_31
    buf31 = reader.storage(None, 2359296)
    reader.tensor(buf31, (768, 768), is_leaf=True)  # primals_32
    buf32 = reader.storage(None, 3072)
    reader.tensor(buf32, (768,), is_leaf=True)  # primals_33
    buf33 = reader.storage(None, 3072)
    reader.tensor(buf33, (768,), is_leaf=True)  # primals_34
    buf34 = reader.storage(None, 3072)
    reader.tensor(buf34, (768,), is_leaf=True)  # primals_35
    buf35 = reader.storage(None, 9437184)
    reader.tensor(buf35, (3072, 768), is_leaf=True)  # primals_36
    buf36 = reader.storage(None, 12288)
    reader.tensor(buf36, (3072,), is_leaf=True)  # primals_37
    buf37 = reader.storage(None, 9437184)
    reader.tensor(buf37, (768, 3072), is_leaf=True)  # primals_38
    buf38 = reader.storage(None, 3072)
    reader.tensor(buf38, (768,), is_leaf=True)  # primals_39
    buf39 = reader.storage(None, 3072)
    reader.tensor(buf39, (768,), is_leaf=True)  # primals_40
    buf40 = reader.storage(None, 3072)
    reader.tensor(buf40, (768,), is_leaf=True)  # primals_41
    buf41 = reader.storage(None, 2359296)
    reader.tensor(buf41, (768, 768), is_leaf=True)  # primals_42
    buf42 = reader.storage(None, 3072)
    reader.tensor(buf42, (768,), is_leaf=True)  # primals_43
    buf43 = reader.storage(None, 2359296)
    reader.tensor(buf43, (768, 768), is_leaf=True)  # primals_44
    buf44 = reader.storage(None, 3072)
    reader.tensor(buf44, (768,), is_leaf=True)  # primals_45
    buf45 = reader.storage(None, 2359296)
    reader.tensor(buf45, (768, 768), is_leaf=True)  # primals_46
    buf46 = reader.storage(None, 3072)
    reader.tensor(buf46, (768,), is_leaf=True)  # primals_47
    buf47 = reader.storage(None, 2359296)
    reader.tensor(buf47, (768, 768), is_leaf=True)  # primals_48
    buf48 = reader.storage(None, 3072)
    reader.tensor(buf48, (768,), is_leaf=True)  # primals_49
    buf49 = reader.storage(None, 3072)
    reader.tensor(buf49, (768,), is_leaf=True)  # primals_50
    buf50 = reader.storage(None, 3072)
    reader.tensor(buf50, (768,), is_leaf=True)  # primals_51
    buf51 = reader.storage(None, 9437184)
    reader.tensor(buf51, (3072, 768), is_leaf=True)  # primals_52
    buf52 = reader.storage(None, 12288)
    reader.tensor(buf52, (3072,), is_leaf=True)  # primals_53
    buf53 = reader.storage(None, 9437184)
    reader.tensor(buf53, (768, 3072), is_leaf=True)  # primals_54
    buf54 = reader.storage(None, 3072)
    reader.tensor(buf54, (768,), is_leaf=True)  # primals_55
    buf55 = reader.storage(None, 3072)
    reader.tensor(buf55, (768,), is_leaf=True)  # primals_56
    buf56 = reader.storage(None, 3072)
    reader.tensor(buf56, (768,), is_leaf=True)  # primals_57
    buf57 = reader.storage(None, 2359296)
    reader.tensor(buf57, (768, 768), is_leaf=True)  # primals_58
    buf58 = reader.storage(None, 3072)
    reader.tensor(buf58, (768,), is_leaf=True)  # primals_59
    buf59 = reader.storage(None, 2359296)
    reader.tensor(buf59, (768, 768), is_leaf=True)  # primals_60
    buf60 = reader.storage(None, 3072)
    reader.tensor(buf60, (768,), is_leaf=True)  # primals_61
    buf61 = reader.storage(None, 2359296)
    reader.tensor(buf61, (768, 768), is_leaf=True)  # primals_62
    buf62 = reader.storage(None, 3072)
    reader.tensor(buf62, (768,), is_leaf=True)  # primals_63
    buf63 = reader.storage(None, 2359296)
    reader.tensor(buf63, (768, 768), is_leaf=True)  # primals_64
    buf64 = reader.storage(None, 3072)
    reader.tensor(buf64, (768,), is_leaf=True)  # primals_65
    buf65 = reader.storage(None, 3072)
    reader.tensor(buf65, (768,), is_leaf=True)  # primals_66
    buf66 = reader.storage(None, 3072)
    reader.tensor(buf66, (768,), is_leaf=True)  # primals_67
    buf67 = reader.storage(None, 9437184)
    reader.tensor(buf67, (3072, 768), is_leaf=True)  # primals_68
    buf68 = reader.storage(None, 12288)
    reader.tensor(buf68, (3072,), is_leaf=True)  # primals_69
    buf69 = reader.storage(None, 9437184)
    reader.tensor(buf69, (768, 3072), is_leaf=True)  # primals_70
    buf70 = reader.storage(None, 3072)
    reader.tensor(buf70, (768,), is_leaf=True)  # primals_71
    buf71 = reader.storage(None, 3072)
    reader.tensor(buf71, (768,), is_leaf=True)  # primals_72
    buf72 = reader.storage(None, 3072)
    reader.tensor(buf72, (768,), is_leaf=True)  # primals_73
    buf73 = reader.storage(None, 2359296)
    reader.tensor(buf73, (768, 768), is_leaf=True)  # primals_74
    buf74 = reader.storage(None, 3072)
    reader.tensor(buf74, (768,), is_leaf=True)  # primals_75
    buf75 = reader.storage(None, 2359296)
    reader.tensor(buf75, (768, 768), is_leaf=True)  # primals_76
    buf76 = reader.storage(None, 3072)
    reader.tensor(buf76, (768,), is_leaf=True)  # primals_77
    buf77 = reader.storage(None, 2359296)
    reader.tensor(buf77, (768, 768), is_leaf=True)  # primals_78
    buf78 = reader.storage(None, 3072)
    reader.tensor(buf78, (768,), is_leaf=True)  # primals_79
    buf79 = reader.storage(None, 2359296)
    reader.tensor(buf79, (768, 768), is_leaf=True)  # primals_80
    buf80 = reader.storage(None, 3072)
    reader.tensor(buf80, (768,), is_leaf=True)  # primals_81
    buf81 = reader.storage(None, 3072)
    reader.tensor(buf81, (768,), is_leaf=True)  # primals_82
    buf82 = reader.storage(None, 3072)
    reader.tensor(buf82, (768,), is_leaf=True)  # primals_83
    buf83 = reader.storage(None, 9437184)
    reader.tensor(buf83, (3072, 768), is_leaf=True)  # primals_84
    buf84 = reader.storage(None, 12288)
    reader.tensor(buf84, (3072,), is_leaf=True)  # primals_85
    buf85 = reader.storage(None, 9437184)
    reader.tensor(buf85, (768, 3072), is_leaf=True)  # primals_86
    buf86 = reader.storage(None, 3072)
    reader.tensor(buf86, (768,), is_leaf=True)  # primals_87
    buf87 = reader.storage(None, 3072)
    reader.tensor(buf87, (768,), is_leaf=True)  # primals_88
    buf88 = reader.storage(None, 3072)
    reader.tensor(buf88, (768,), is_leaf=True)  # primals_89
    buf89 = reader.storage(None, 2359296)
    reader.tensor(buf89, (768, 768), is_leaf=True)  # primals_90
    buf90 = reader.storage(None, 3072)
    reader.tensor(buf90, (768,), is_leaf=True)  # primals_91
    buf91 = reader.storage(None, 2359296)
    reader.tensor(buf91, (768, 768), is_leaf=True)  # primals_92
    buf92 = reader.storage(None, 3072)
    reader.tensor(buf92, (768,), is_leaf=True)  # primals_93
    buf93 = reader.storage(None, 2359296)
    reader.tensor(buf93, (768, 768), is_leaf=True)  # primals_94
    buf94 = reader.storage(None, 3072)
    reader.tensor(buf94, (768,), is_leaf=True)  # primals_95
    buf95 = reader.storage(None, 2359296)
    reader.tensor(buf95, (768, 768), is_leaf=True)  # primals_96
    buf96 = reader.storage(None, 3072)
    reader.tensor(buf96, (768,), is_leaf=True)  # primals_97
    buf97 = reader.storage(None, 3072)
    reader.tensor(buf97, (768,), is_leaf=True)  # primals_98
    buf98 = reader.storage(None, 3072)
    reader.tensor(buf98, (768,), is_leaf=True)  # primals_99
    buf99 = reader.storage(None, 9437184)
    reader.tensor(buf99, (3072, 768), is_leaf=True)  # primals_100
    buf100 = reader.storage(None, 12288)
    reader.tensor(buf100, (3072,), is_leaf=True)  # primals_101
    buf101 = reader.storage(None, 9437184)
    reader.tensor(buf101, (768, 3072), is_leaf=True)  # primals_102
    buf102 = reader.storage(None, 3072)
    reader.tensor(buf102, (768,), is_leaf=True)  # primals_103
    buf103 = reader.storage(None, 3072)
    reader.tensor(buf103, (768,), is_leaf=True)  # primals_104
    buf104 = reader.storage(None, 3072)
    reader.tensor(buf104, (768,), is_leaf=True)  # primals_105
    buf105 = reader.storage(None, 2359296)
    reader.tensor(buf105, (768, 768), is_leaf=True)  # primals_106
    buf106 = reader.storage(None, 3072)
    reader.tensor(buf106, (768,), is_leaf=True)  # primals_107
    buf107 = reader.storage(None, 2359296)
    reader.tensor(buf107, (768, 768), is_leaf=True)  # primals_108
    buf108 = reader.storage(None, 3072)
    reader.tensor(buf108, (768,), is_leaf=True)  # primals_109
    buf109 = reader.storage(None, 2359296)
    reader.tensor(buf109, (768, 768), is_leaf=True)  # primals_110
    buf110 = reader.storage(None, 3072)
    reader.tensor(buf110, (768,), is_leaf=True)  # primals_111
    buf111 = reader.storage(None, 2359296)
    reader.tensor(buf111, (768, 768), is_leaf=True)  # primals_112
    buf112 = reader.storage(None, 3072)
    reader.tensor(buf112, (768,), is_leaf=True)  # primals_113
    buf113 = reader.storage(None, 3072)
    reader.tensor(buf113, (768,), is_leaf=True)  # primals_114
    buf114 = reader.storage(None, 3072)
    reader.tensor(buf114, (768,), is_leaf=True)  # primals_115
    buf115 = reader.storage(None, 9437184)
    reader.tensor(buf115, (3072, 768), is_leaf=True)  # primals_116
    buf116 = reader.storage(None, 12288)
    reader.tensor(buf116, (3072,), is_leaf=True)  # primals_117
    buf117 = reader.storage(None, 9437184)
    reader.tensor(buf117, (768, 3072), is_leaf=True)  # primals_118
    buf118 = reader.storage(None, 3072)
    reader.tensor(buf118, (768,), is_leaf=True)  # primals_119
    buf119 = reader.storage(None, 3072)
    reader.tensor(buf119, (768,), is_leaf=True)  # primals_120
    buf120 = reader.storage(None, 3072)
    reader.tensor(buf120, (768,), is_leaf=True)  # primals_121
    buf121 = reader.storage(None, 2359296)
    reader.tensor(buf121, (768, 768), is_leaf=True)  # primals_122
    buf122 = reader.storage(None, 3072)
    reader.tensor(buf122, (768,), is_leaf=True)  # primals_123
    buf123 = reader.storage(None, 2359296)
    reader.tensor(buf123, (768, 768), is_leaf=True)  # primals_124
    buf124 = reader.storage(None, 3072)
    reader.tensor(buf124, (768,), is_leaf=True)  # primals_125
    buf125 = reader.storage(None, 2359296)
    reader.tensor(buf125, (768, 768), is_leaf=True)  # primals_126
    buf126 = reader.storage(None, 3072)
    reader.tensor(buf126, (768,), is_leaf=True)  # primals_127
    buf127 = reader.storage(None, 2359296)
    reader.tensor(buf127, (768, 768), is_leaf=True)  # primals_128
    buf128 = reader.storage(None, 3072)
    reader.tensor(buf128, (768,), is_leaf=True)  # primals_129
    buf129 = reader.storage(None, 3072)
    reader.tensor(buf129, (768,), is_leaf=True)  # primals_130
    buf130 = reader.storage(None, 3072)
    reader.tensor(buf130, (768,), is_leaf=True)  # primals_131
    buf131 = reader.storage(None, 9437184)
    reader.tensor(buf131, (3072, 768), is_leaf=True)  # primals_132
    buf132 = reader.storage(None, 12288)
    reader.tensor(buf132, (3072,), is_leaf=True)  # primals_133
    buf133 = reader.storage(None, 9437184)
    reader.tensor(buf133, (768, 3072), is_leaf=True)  # primals_134
    buf134 = reader.storage(None, 3072)
    reader.tensor(buf134, (768,), is_leaf=True)  # primals_135
    buf135 = reader.storage(None, 3072)
    reader.tensor(buf135, (768,), is_leaf=True)  # primals_136
    buf136 = reader.storage(None, 3072)
    reader.tensor(buf136, (768,), is_leaf=True)  # primals_137
    buf137 = reader.storage(None, 2359296)
    reader.tensor(buf137, (768, 768), is_leaf=True)  # primals_138
    buf138 = reader.storage(None, 3072)
    reader.tensor(buf138, (768,), is_leaf=True)  # primals_139
    buf139 = reader.storage(None, 2359296)
    reader.tensor(buf139, (768, 768), is_leaf=True)  # primals_140
    buf140 = reader.storage(None, 3072)
    reader.tensor(buf140, (768,), is_leaf=True)  # primals_141
    buf141 = reader.storage(None, 2359296)
    reader.tensor(buf141, (768, 768), is_leaf=True)  # primals_142
    buf142 = reader.storage(None, 3072)
    reader.tensor(buf142, (768,), is_leaf=True)  # primals_143
    buf143 = reader.storage(None, 2359296)
    reader.tensor(buf143, (768, 768), is_leaf=True)  # primals_144
    buf144 = reader.storage(None, 3072)
    reader.tensor(buf144, (768,), is_leaf=True)  # primals_145
    buf145 = reader.storage(None, 3072)
    reader.tensor(buf145, (768,), is_leaf=True)  # primals_146
    buf146 = reader.storage(None, 3072)
    reader.tensor(buf146, (768,), is_leaf=True)  # primals_147
    buf147 = reader.storage(None, 9437184)
    reader.tensor(buf147, (3072, 768), is_leaf=True)  # primals_148
    buf148 = reader.storage(None, 12288)
    reader.tensor(buf148, (3072,), is_leaf=True)  # primals_149
    buf149 = reader.storage(None, 9437184)
    reader.tensor(buf149, (768, 3072), is_leaf=True)  # primals_150
    buf150 = reader.storage(None, 3072)
    reader.tensor(buf150, (768,), is_leaf=True)  # primals_151
    buf151 = reader.storage(None, 3072)
    reader.tensor(buf151, (768,), is_leaf=True)  # primals_152
    buf152 = reader.storage(None, 3072)
    reader.tensor(buf152, (768,), is_leaf=True)  # primals_153
    buf153 = reader.storage(None, 2359296)
    reader.tensor(buf153, (768, 768), is_leaf=True)  # primals_154
    buf154 = reader.storage(None, 3072)
    reader.tensor(buf154, (768,), is_leaf=True)  # primals_155
    buf155 = reader.storage(None, 2359296)
    reader.tensor(buf155, (768, 768), is_leaf=True)  # primals_156
    buf156 = reader.storage(None, 3072)
    reader.tensor(buf156, (768,), is_leaf=True)  # primals_157
    buf157 = reader.storage(None, 2359296)
    reader.tensor(buf157, (768, 768), is_leaf=True)  # primals_158
    buf158 = reader.storage(None, 3072)
    reader.tensor(buf158, (768,), is_leaf=True)  # primals_159
    buf159 = reader.storage(None, 2359296)
    reader.tensor(buf159, (768, 768), is_leaf=True)  # primals_160
    buf160 = reader.storage(None, 3072)
    reader.tensor(buf160, (768,), is_leaf=True)  # primals_161
    buf161 = reader.storage(None, 3072)
    reader.tensor(buf161, (768,), is_leaf=True)  # primals_162
    buf162 = reader.storage(None, 3072)
    reader.tensor(buf162, (768,), is_leaf=True)  # primals_163
    buf163 = reader.storage(None, 9437184)
    reader.tensor(buf163, (3072, 768), is_leaf=True)  # primals_164
    buf164 = reader.storage(None, 12288)
    reader.tensor(buf164, (3072,), is_leaf=True)  # primals_165
    buf165 = reader.storage(None, 9437184)
    reader.tensor(buf165, (768, 3072), is_leaf=True)  # primals_166
    buf166 = reader.storage(None, 3072)
    reader.tensor(buf166, (768,), is_leaf=True)  # primals_167
    buf167 = reader.storage(None, 3072)
    reader.tensor(buf167, (768,), is_leaf=True)  # primals_168
    buf168 = reader.storage(None, 3072)
    reader.tensor(buf168, (768,), is_leaf=True)  # primals_169
    buf169 = reader.storage(None, 2359296)
    reader.tensor(buf169, (768, 768), is_leaf=True)  # primals_170
    buf170 = reader.storage(None, 3072)
    reader.tensor(buf170, (768,), is_leaf=True)  # primals_171
    buf171 = reader.storage(None, 2359296)
    reader.tensor(buf171, (768, 768), is_leaf=True)  # primals_172
    buf172 = reader.storage(None, 3072)
    reader.tensor(buf172, (768,), is_leaf=True)  # primals_173
    buf173 = reader.storage(None, 2359296)
    reader.tensor(buf173, (768, 768), is_leaf=True)  # primals_174
    buf174 = reader.storage(None, 3072)
    reader.tensor(buf174, (768,), is_leaf=True)  # primals_175
    buf175 = reader.storage(None, 2359296)
    reader.tensor(buf175, (768, 768), is_leaf=True)  # primals_176
    buf176 = reader.storage(None, 3072)
    reader.tensor(buf176, (768,), is_leaf=True)  # primals_177
    buf177 = reader.storage(None, 3072)
    reader.tensor(buf177, (768,), is_leaf=True)  # primals_178
    buf178 = reader.storage(None, 3072)
    reader.tensor(buf178, (768,), is_leaf=True)  # primals_179
    buf179 = reader.storage(None, 9437184)
    reader.tensor(buf179, (3072, 768), is_leaf=True)  # primals_180
    buf180 = reader.storage(None, 12288)
    reader.tensor(buf180, (3072,), is_leaf=True)  # primals_181
    buf181 = reader.storage(None, 9437184)
    reader.tensor(buf181, (768, 3072), is_leaf=True)  # primals_182
    buf182 = reader.storage(None, 3072)
    reader.tensor(buf182, (768,), is_leaf=True)  # primals_183
    buf183 = reader.storage(None, 3072)
    reader.tensor(buf183, (768,), is_leaf=True)  # primals_184
    buf184 = reader.storage(None, 3072)
    reader.tensor(buf184, (768,), is_leaf=True)  # primals_185
    buf185 = reader.storage(None, 2359296)
    reader.tensor(buf185, (768, 768), is_leaf=True)  # primals_186
    buf186 = reader.storage(None, 3072)
    reader.tensor(buf186, (768,), is_leaf=True)  # primals_187
    buf187 = reader.storage(None, 2359296)
    reader.tensor(buf187, (768, 768), is_leaf=True)  # primals_188
    buf188 = reader.storage(None, 3072)
    reader.tensor(buf188, (768,), is_leaf=True)  # primals_189
    buf189 = reader.storage(None, 2359296)
    reader.tensor(buf189, (768, 768), is_leaf=True)  # primals_190
    buf190 = reader.storage(None, 3072)
    reader.tensor(buf190, (768,), is_leaf=True)  # primals_191
    buf191 = reader.storage(None, 2359296)
    reader.tensor(buf191, (768, 768), is_leaf=True)  # primals_192
    buf192 = reader.storage(None, 3072)
    reader.tensor(buf192, (768,), is_leaf=True)  # primals_193
    buf193 = reader.storage(None, 3072)
    reader.tensor(buf193, (768,), is_leaf=True)  # primals_194
    buf194 = reader.storage(None, 3072)
    reader.tensor(buf194, (768,), is_leaf=True)  # primals_195
    buf195 = reader.storage(None, 9437184)
    reader.tensor(buf195, (3072, 768), is_leaf=True)  # primals_196
    buf196 = reader.storage(None, 12288)
    reader.tensor(buf196, (3072,), is_leaf=True)  # primals_197
    buf197 = reader.storage(None, 9437184)
    reader.tensor(buf197, (768, 3072), is_leaf=True)  # primals_198
    buf198 = reader.storage(None, 3072)
    reader.tensor(buf198, (768,), is_leaf=True)  # primals_199
    buf199 = reader.storage(None, 3072)
    reader.tensor(buf199, (768,), is_leaf=True)  # primals_200
    buf200 = reader.storage(None, 3072)
    reader.tensor(buf200, (768,), is_leaf=True)  # primals_201
    buf201 = reader.storage(None, 2359296)
    reader.tensor(buf201, (768, 768), is_leaf=True)  # primals_202
    buf202 = reader.storage(None, 3072)
    reader.tensor(buf202, (768,), is_leaf=True)  # primals_203
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)