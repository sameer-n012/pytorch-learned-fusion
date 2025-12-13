class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 8]", primals_2: "i64[1, 512]", primals_3: "i64[1, 512]", primals_4: "f32[30522, 768]", primals_5: "f32[2, 768]", primals_6: "f32[512, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "i64[1, 8]", primals_10: "f32[768, 768]", primals_11: "f32[768]", primals_12: "f32[768, 768]", primals_13: "f32[768]", primals_14: "f32[768, 768]", primals_15: "f32[768]", primals_16: "f32[768, 768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[768]", primals_20: "f32[3072, 768]", primals_21: "f32[3072]", primals_22: "f32[768, 3072]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768, 768]", primals_27: "f32[768]", primals_28: "f32[768, 768]", primals_29: "f32[768]", primals_30: "f32[768, 768]", primals_31: "f32[768]", primals_32: "f32[768, 768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[3072, 768]", primals_37: "f32[3072]", primals_38: "f32[768, 3072]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768, 768]", primals_43: "f32[768]", primals_44: "f32[768, 768]", primals_45: "f32[768]", primals_46: "f32[768, 768]", primals_47: "f32[768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[3072, 768]", primals_53: "f32[3072]", primals_54: "f32[768, 3072]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[768, 768]", primals_59: "f32[768]", primals_60: "f32[768, 768]", primals_61: "f32[768]", primals_62: "f32[768, 768]", primals_63: "f32[768]", primals_64: "f32[768, 768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[768]", primals_68: "f32[3072, 768]", primals_69: "f32[3072]", primals_70: "f32[768, 3072]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768, 768]", primals_75: "f32[768]", primals_76: "f32[768, 768]", primals_77: "f32[768]", primals_78: "f32[768, 768]", primals_79: "f32[768]", primals_80: "f32[768, 768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[3072, 768]", primals_85: "f32[3072]", primals_86: "f32[768, 3072]", primals_87: "f32[768]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768, 768]", primals_91: "f32[768]", primals_92: "f32[768, 768]", primals_93: "f32[768]", primals_94: "f32[768, 768]", primals_95: "f32[768]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[3072, 768]", primals_101: "f32[3072]", primals_102: "f32[768, 3072]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768, 768]", primals_107: "f32[768]", primals_108: "f32[768, 768]", primals_109: "f32[768]", primals_110: "f32[768, 768]", primals_111: "f32[768]", primals_112: "f32[768, 768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[3072, 768]", primals_117: "f32[3072]", primals_118: "f32[768, 3072]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768]", primals_124: "f32[768, 768]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[768, 768]", primals_129: "f32[768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[3072, 768]", primals_133: "f32[3072]", primals_134: "f32[768, 3072]", primals_135: "f32[768]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[768, 768]", primals_141: "f32[768]", primals_142: "f32[768, 768]", primals_143: "f32[768]", primals_144: "f32[768, 768]", primals_145: "f32[768]", primals_146: "f32[768]", primals_147: "f32[768]", primals_148: "f32[3072, 768]", primals_149: "f32[3072]", primals_150: "f32[768, 3072]", primals_151: "f32[768]", primals_152: "f32[768]", primals_153: "f32[768]", primals_154: "f32[768, 768]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[768, 768]", primals_159: "f32[768]", primals_160: "f32[768, 768]", primals_161: "f32[768]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[3072, 768]", primals_165: "f32[3072]", primals_166: "f32[768, 3072]", primals_167: "f32[768]", primals_168: "f32[768]", primals_169: "f32[768]", primals_170: "f32[768, 768]", primals_171: "f32[768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[768, 768]", primals_177: "f32[768]", primals_178: "f32[768]", primals_179: "f32[768]", primals_180: "f32[3072, 768]", primals_181: "f32[3072]", primals_182: "f32[768, 3072]", primals_183: "f32[768]", primals_184: "f32[768]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[768, 768]", primals_189: "f32[768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[768, 768]", primals_193: "f32[768]", primals_194: "f32[768]", primals_195: "f32[768]", primals_196: "f32[3072, 768]", primals_197: "f32[3072]", primals_198: "f32[768, 3072]", primals_199: "f32[768]", primals_200: "f32[768]", primals_201: "f32[768]", primals_202: "f32[768, 768]", primals_203: "f32[768]"):
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:930 in forward, code: buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
        slice_1: "i64[1, 8]" = torch.ops.aten.slice.Tensor(primals_2, 1, 0, 8)
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:931 in forward, code: buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        expand: "i64[1, 8]" = torch.ops.aten.expand.default(slice_1, [1, 8]);  slice_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:165 in forward, code: position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
        slice_2: "i64[1, 8]" = torch.ops.aten.slice.Tensor(primals_3, 1, 0, 8)
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:179 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[1, 8, 768]" = torch.ops.aten.embedding.default(primals_4, primals_1, 0);  primals_4 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:180 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[1, 8, 768]" = torch.ops.aten.embedding.default(primals_5, expand);  primals_5 = expand = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:182 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:184 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 8, 768]" = torch.ops.aten.embedding.default(primals_6, slice_2);  primals_6 = slice_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:185 in forward, code: embeddings += position_embeddings
        add_1: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:186 in forward, code: embeddings = self.LayerNorm(embeddings)
        var_mean = torch.ops.aten.var_mean.correction(add_1, [2], correction = 0, keepdim = True)
        getitem: "f32[1, 8, 1]" = var_mean[0]
        getitem_1: "f32[1, 8, 1]" = var_mean[1];  var_mean = None
        add_2: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_1, getitem_1);  add_1 = getitem_1 = None
        mul: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul, primals_7)
        add_3: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_8);  mul_1 = primals_8 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:194 in _expand_mask, code: expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        unsqueeze: "i64[1, 1, 8]" = torch.ops.aten.unsqueeze.default(primals_9, 1);  primals_9 = None
        unsqueeze_1: "i64[1, 1, 1, 8]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        expand_1: "i64[1, 1, 8, 8]" = torch.ops.aten.expand.default(unsqueeze_1, [1, 1, 8, 8]);  unsqueeze_1 = None
        convert_element_type: "f32[1, 1, 8, 8]" = torch.ops.prims.convert_element_type.default(expand_1, torch.float32);  expand_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:196 in _expand_mask, code: inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask
        full_default: "f32[]" = torch.ops.aten.full.default([], 1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        sub_1: "f32[1, 1, 8, 8]" = torch.ops.aten.sub.Tensor(full_default, convert_element_type);  full_default = convert_element_type = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/modeling_attn_mask_utils.py:198 in _expand_mask, code: return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        convert_element_type_1: "b8[1, 1, 8, 8]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bool)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: "f32[1, 1, 8, 8]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, sub_1);  convert_element_type_1 = full_default_1 = sub_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view: "f32[8, 768]" = torch.ops.aten.view.default(add_3, [8, 768])
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0])
        addmm: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_11, view, permute);  primals_11 = permute = None
        view_1: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm, [1, 8, 768]);  addmm = None
        view_2: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_1, [1, -1, 12, 64]);  view_1 = None
        permute_1: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_12, [1, 0])
        addmm_1: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_13, view, permute_2);  primals_13 = permute_2 = None
        view_4: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_1, [1, 8, 768]);  addmm_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_5: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_4, [1, -1, 12, 64]);  view_4 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_3: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0])
        addmm_2: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_15, view, permute_4);  primals_15 = permute_4 = None
        view_7: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_2, [1, 8, 768]);  addmm_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_8: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_7, [1, -1, 12, 64]);  view_7 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_5: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_1, permute_3, permute_5, attn_mask = where)
        getitem_2: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu[0]
        getitem_3: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu[1];  _scaled_dot_product_flash_attention_for_cpu = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_6: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_9: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_6, [1, 8, 768]);  permute_6 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_10: "f32[8, 768]" = torch.ops.aten.view.default(view_9, [8, 768]);  view_9 = None
        permute_7: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0])
        addmm_3: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_17, view_10, permute_7);  primals_17 = view_10 = permute_7 = None
        view_11: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_3, [1, 8, 768]);  addmm_3 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_4: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_11, add_3);  view_11 = add_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem_4: "f32[1, 8, 1]" = var_mean_1[0]
        getitem_5: "f32[1, 8, 1]" = var_mean_1[1];  var_mean_1 = None
        add_5: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_1: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_4, getitem_5);  add_4 = getitem_5 = None
        mul_2: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_3: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_18)
        add_6: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_19);  mul_3 = primals_19 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_12: "f32[8, 768]" = torch.ops.aten.view.default(add_6, [8, 768])
        permute_8: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_20, [1, 0])
        addmm_4: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_21, view_12, permute_8);  primals_21 = permute_8 = None
        view_13: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_4: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
        mul_5: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
        erf: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_7: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_14: "f32[8, 3072]" = torch.ops.aten.view.default(mul_6, [8, 3072]);  mul_6 = None
        permute_9: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0])
        addmm_5: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_23, view_14, permute_9);  primals_23 = permute_9 = None
        view_15: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_5, [1, 8, 768]);  addmm_5 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_8: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_15, add_6);  view_15 = add_6 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_6: "f32[1, 8, 1]" = var_mean_2[0]
        getitem_7: "f32[1, 8, 1]" = var_mean_2[1];  var_mean_2 = None
        add_9: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_2: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_8, getitem_7);  add_8 = getitem_7 = None
        mul_7: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_8: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_24)
        add_10: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_25);  mul_8 = primals_25 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_16: "f32[8, 768]" = torch.ops.aten.view.default(add_10, [8, 768])
        permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_26, [1, 0])
        addmm_6: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_27, view_16, permute_10);  primals_27 = permute_10 = None
        view_17: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_6, [1, 8, 768]);  addmm_6 = None
        view_18: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_17, [1, -1, 12, 64]);  view_17 = None
        permute_11: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0])
        addmm_7: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_29, view_16, permute_12);  primals_29 = permute_12 = None
        view_20: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_7, [1, 8, 768]);  addmm_7 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_21: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_20, [1, -1, 12, 64]);  view_20 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_13: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0])
        addmm_8: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_31, view_16, permute_14);  primals_31 = permute_14 = None
        view_23: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_8, [1, 8, 768]);  addmm_8 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_24: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_23, [1, -1, 12, 64]);  view_23 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_15: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_1 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_11, permute_13, permute_15, attn_mask = where)
        getitem_8: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_1[0]
        getitem_9: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_1[1];  _scaled_dot_product_flash_attention_for_cpu_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_16: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_25: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_16, [1, 8, 768]);  permute_16 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_26: "f32[8, 768]" = torch.ops.aten.view.default(view_25, [8, 768]);  view_25 = None
        permute_17: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0])
        addmm_9: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_33, view_26, permute_17);  primals_33 = view_26 = permute_17 = None
        view_27: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_9, [1, 8, 768]);  addmm_9 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_11: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_27, add_10);  view_27 = add_10 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
        getitem_10: "f32[1, 8, 1]" = var_mean_3[0]
        getitem_11: "f32[1, 8, 1]" = var_mean_3[1];  var_mean_3 = None
        add_12: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_3: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_4: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_11);  add_11 = getitem_11 = None
        mul_9: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
        mul_10: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_34)
        add_13: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_35);  mul_10 = primals_35 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_28: "f32[8, 768]" = torch.ops.aten.view.default(add_13, [8, 768])
        permute_18: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_36, [1, 0])
        addmm_10: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_37, view_28, permute_18);  primals_37 = permute_18 = None
        view_29: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_11: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.5)
        mul_12: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_29, 0.7071067811865476);  view_29 = None
        erf_1: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_14: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_14);  mul_11 = add_14 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_30: "f32[8, 3072]" = torch.ops.aten.view.default(mul_13, [8, 3072]);  mul_13 = None
        permute_19: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0])
        addmm_11: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_39, view_30, permute_19);  primals_39 = permute_19 = None
        view_31: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_11, [1, 8, 768]);  addmm_11 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_15: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_31, add_13);  view_31 = add_13 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_12: "f32[1, 8, 1]" = var_mean_4[0]
        getitem_13: "f32[1, 8, 1]" = var_mean_4[1];  var_mean_4 = None
        add_16: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_4: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_5: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_13);  add_15 = getitem_13 = None
        mul_14: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = None
        mul_15: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_40)
        add_17: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_41);  mul_15 = primals_41 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_32: "f32[8, 768]" = torch.ops.aten.view.default(add_17, [8, 768])
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_42, [1, 0])
        addmm_12: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_43, view_32, permute_20);  primals_43 = permute_20 = None
        view_33: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_12, [1, 8, 768]);  addmm_12 = None
        view_34: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_33, [1, -1, 12, 64]);  view_33 = None
        permute_21: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_44, [1, 0])
        addmm_13: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_45, view_32, permute_22);  primals_45 = permute_22 = None
        view_36: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_13, [1, 8, 768]);  addmm_13 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_37: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_36, [1, -1, 12, 64]);  view_36 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_23: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3]);  view_37 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0])
        addmm_14: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_47, view_32, permute_24);  primals_47 = permute_24 = None
        view_39: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_14, [1, 8, 768]);  addmm_14 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_40: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_39, [1, -1, 12, 64]);  view_39 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_25: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_2 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_21, permute_23, permute_25, attn_mask = where)
        getitem_14: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_2[0]
        getitem_15: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_2[1];  _scaled_dot_product_flash_attention_for_cpu_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_26: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_14, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_41: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_26, [1, 8, 768]);  permute_26 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_42: "f32[8, 768]" = torch.ops.aten.view.default(view_41, [8, 768]);  view_41 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0])
        addmm_15: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_49, view_42, permute_27);  primals_49 = view_42 = permute_27 = None
        view_43: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_15, [1, 8, 768]);  addmm_15 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_18: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_43, add_17);  view_43 = add_17 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_16: "f32[1, 8, 1]" = var_mean_5[0]
        getitem_17: "f32[1, 8, 1]" = var_mean_5[1];  var_mean_5 = None
        add_19: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_5: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        sub_6: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_18, getitem_17);  add_18 = getitem_17 = None
        mul_16: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = None
        mul_17: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_50)
        add_20: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_51);  mul_17 = primals_51 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_44: "f32[8, 768]" = torch.ops.aten.view.default(add_20, [8, 768])
        permute_28: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_52, [1, 0])
        addmm_16: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_53, view_44, permute_28);  primals_53 = permute_28 = None
        view_45: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_18: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.5)
        mul_19: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_45, 0.7071067811865476);  view_45 = None
        erf_2: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_21: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_21);  mul_18 = add_21 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_46: "f32[8, 3072]" = torch.ops.aten.view.default(mul_20, [8, 3072]);  mul_20 = None
        permute_29: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0])
        addmm_17: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_55, view_46, permute_29);  primals_55 = permute_29 = None
        view_47: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_17, [1, 8, 768]);  addmm_17 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_22: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_47, add_20);  view_47 = add_20 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
        getitem_18: "f32[1, 8, 1]" = var_mean_6[0]
        getitem_19: "f32[1, 8, 1]" = var_mean_6[1];  var_mean_6 = None
        add_23: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_6: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_7: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_22, getitem_19);  add_22 = getitem_19 = None
        mul_21: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_6);  sub_7 = None
        mul_22: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_56)
        add_24: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_57);  mul_22 = primals_57 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_48: "f32[8, 768]" = torch.ops.aten.view.default(add_24, [8, 768])
        permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0])
        addmm_18: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_59, view_48, permute_30);  primals_59 = permute_30 = None
        view_49: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_18, [1, 8, 768]);  addmm_18 = None
        view_50: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_49, [1, -1, 12, 64]);  view_49 = None
        permute_31: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_32: "f32[768, 768]" = torch.ops.aten.permute.default(primals_60, [1, 0])
        addmm_19: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_61, view_48, permute_32);  primals_61 = permute_32 = None
        view_52: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_19, [1, 8, 768]);  addmm_19 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_53: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_52, [1, -1, 12, 64]);  view_52 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_33: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0])
        addmm_20: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_63, view_48, permute_34);  primals_63 = permute_34 = None
        view_55: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_20, [1, 8, 768]);  addmm_20 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_56: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_55, [1, -1, 12, 64]);  view_55 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_35: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_3 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_31, permute_33, permute_35, attn_mask = where)
        getitem_20: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_3[0]
        getitem_21: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_3[1];  _scaled_dot_product_flash_attention_for_cpu_3 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_36: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_20, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_57: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_36, [1, 8, 768]);  permute_36 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_58: "f32[8, 768]" = torch.ops.aten.view.default(view_57, [8, 768]);  view_57 = None
        permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0])
        addmm_21: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_65, view_58, permute_37);  primals_65 = view_58 = permute_37 = None
        view_59: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_21, [1, 8, 768]);  addmm_21 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_25: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_59, add_24);  view_59 = add_24 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_22: "f32[1, 8, 1]" = var_mean_7[0]
        getitem_23: "f32[1, 8, 1]" = var_mean_7[1];  var_mean_7 = None
        add_26: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_7: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_8: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_25, getitem_23);  add_25 = getitem_23 = None
        mul_23: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_7);  sub_8 = None
        mul_24: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_66)
        add_27: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_67);  mul_24 = primals_67 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_60: "f32[8, 768]" = torch.ops.aten.view.default(add_27, [8, 768])
        permute_38: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_68, [1, 0])
        addmm_22: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_69, view_60, permute_38);  primals_69 = permute_38 = None
        view_61: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_25: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_26: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476);  view_61 = None
        erf_3: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_28: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_62: "f32[8, 3072]" = torch.ops.aten.view.default(mul_27, [8, 3072]);  mul_27 = None
        permute_39: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0])
        addmm_23: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_71, view_62, permute_39);  primals_71 = permute_39 = None
        view_63: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_23, [1, 8, 768]);  addmm_23 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_29: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_63, add_27);  view_63 = add_27 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_24: "f32[1, 8, 1]" = var_mean_8[0]
        getitem_25: "f32[1, 8, 1]" = var_mean_8[1];  var_mean_8 = None
        add_30: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_8: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_9: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_29, getitem_25);  add_29 = getitem_25 = None
        mul_28: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_8);  sub_9 = None
        mul_29: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_72)
        add_31: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_73);  mul_29 = primals_73 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_64: "f32[8, 768]" = torch.ops.aten.view.default(add_31, [8, 768])
        permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(primals_74, [1, 0])
        addmm_24: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_75, view_64, permute_40);  primals_75 = permute_40 = None
        view_65: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_24, [1, 8, 768]);  addmm_24 = None
        view_66: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_65, [1, -1, 12, 64]);  view_65 = None
        permute_41: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0])
        addmm_25: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_77, view_64, permute_42);  primals_77 = permute_42 = None
        view_68: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_25, [1, 8, 768]);  addmm_25 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_69: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_68, [1, -1, 12, 64]);  view_68 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_43: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0])
        addmm_26: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_79, view_64, permute_44);  primals_79 = permute_44 = None
        view_71: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_26, [1, 8, 768]);  addmm_26 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_72: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_71, [1, -1, 12, 64]);  view_71 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_45: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_4 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_41, permute_43, permute_45, attn_mask = where)
        getitem_26: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_4[0]
        getitem_27: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_4[1];  _scaled_dot_product_flash_attention_for_cpu_4 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_46: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_26, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_73: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_46, [1, 8, 768]);  permute_46 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_74: "f32[8, 768]" = torch.ops.aten.view.default(view_73, [8, 768]);  view_73 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0])
        addmm_27: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_81, view_74, permute_47);  primals_81 = view_74 = permute_47 = None
        view_75: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_27, [1, 8, 768]);  addmm_27 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_32: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_75, add_31);  view_75 = add_31 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
        getitem_28: "f32[1, 8, 1]" = var_mean_9[0]
        getitem_29: "f32[1, 8, 1]" = var_mean_9[1];  var_mean_9 = None
        add_33: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_9: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_10: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_32, getitem_29);  add_32 = getitem_29 = None
        mul_30: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_9);  sub_10 = None
        mul_31: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_82)
        add_34: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_83);  mul_31 = primals_83 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_76: "f32[8, 768]" = torch.ops.aten.view.default(add_34, [8, 768])
        permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_84, [1, 0])
        addmm_28: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_85, view_76, permute_48);  primals_85 = permute_48 = None
        view_77: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_32: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        mul_33: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_77, 0.7071067811865476);  view_77 = None
        erf_4: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_35: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_78: "f32[8, 3072]" = torch.ops.aten.view.default(mul_34, [8, 3072]);  mul_34 = None
        permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0])
        addmm_29: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_87, view_78, permute_49);  primals_87 = permute_49 = None
        view_79: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_29, [1, 8, 768]);  addmm_29 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_36: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_79, add_34);  view_79 = add_34 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
        getitem_30: "f32[1, 8, 1]" = var_mean_10[0]
        getitem_31: "f32[1, 8, 1]" = var_mean_10[1];  var_mean_10 = None
        add_37: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_10: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_11: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_36, getitem_31);  add_36 = getitem_31 = None
        mul_35: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_10);  sub_11 = None
        mul_36: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_88)
        add_38: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_89);  mul_36 = primals_89 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_80: "f32[8, 768]" = torch.ops.aten.view.default(add_38, [8, 768])
        permute_50: "f32[768, 768]" = torch.ops.aten.permute.default(primals_90, [1, 0])
        addmm_30: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_91, view_80, permute_50);  primals_91 = permute_50 = None
        view_81: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_30, [1, 8, 768]);  addmm_30 = None
        view_82: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_81, [1, -1, 12, 64]);  view_81 = None
        permute_51: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_92, [1, 0])
        addmm_31: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_93, view_80, permute_52);  primals_93 = permute_52 = None
        view_84: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_31, [1, 8, 768]);  addmm_31 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_85: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_84, [1, -1, 12, 64]);  view_84 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_53: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0])
        addmm_32: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_95, view_80, permute_54);  primals_95 = permute_54 = None
        view_87: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_32, [1, 8, 768]);  addmm_32 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_88: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_87, [1, -1, 12, 64]);  view_87 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_55: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_5 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_51, permute_53, permute_55, attn_mask = where)
        getitem_32: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_5[0]
        getitem_33: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_5[1];  _scaled_dot_product_flash_attention_for_cpu_5 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_56: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_89: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_56, [1, 8, 768]);  permute_56 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_90: "f32[8, 768]" = torch.ops.aten.view.default(view_89, [8, 768]);  view_89 = None
        permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0])
        addmm_33: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_97, view_90, permute_57);  primals_97 = view_90 = permute_57 = None
        view_91: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_33, [1, 8, 768]);  addmm_33 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_39: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_91, add_38);  view_91 = add_38 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
        getitem_34: "f32[1, 8, 1]" = var_mean_11[0]
        getitem_35: "f32[1, 8, 1]" = var_mean_11[1];  var_mean_11 = None
        add_40: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_11: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        sub_12: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_35);  add_39 = getitem_35 = None
        mul_37: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_11);  sub_12 = None
        mul_38: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_98)
        add_41: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_99);  mul_38 = primals_99 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_92: "f32[8, 768]" = torch.ops.aten.view.default(add_41, [8, 768])
        permute_58: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_100, [1, 0])
        addmm_34: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_101, view_92, permute_58);  primals_101 = permute_58 = None
        view_93: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_39: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.5)
        mul_40: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_93, 0.7071067811865476);  view_93 = None
        erf_5: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_42: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_94: "f32[8, 3072]" = torch.ops.aten.view.default(mul_41, [8, 3072]);  mul_41 = None
        permute_59: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_102, [1, 0])
        addmm_35: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_103, view_94, permute_59);  primals_103 = permute_59 = None
        view_95: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_35, [1, 8, 768]);  addmm_35 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_43: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_95, add_41);  view_95 = add_41 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
        getitem_36: "f32[1, 8, 1]" = var_mean_12[0]
        getitem_37: "f32[1, 8, 1]" = var_mean_12[1];  var_mean_12 = None
        add_44: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_12: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_13: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_37);  add_43 = getitem_37 = None
        mul_42: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_12);  sub_13 = None
        mul_43: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_104)
        add_45: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_105);  mul_43 = primals_105 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_96: "f32[8, 768]" = torch.ops.aten.view.default(add_45, [8, 768])
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0])
        addmm_36: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_107, view_96, permute_60);  primals_107 = permute_60 = None
        view_97: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_36, [1, 8, 768]);  addmm_36 = None
        view_98: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_97, [1, -1, 12, 64]);  view_97 = None
        permute_61: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(primals_108, [1, 0])
        addmm_37: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_109, view_96, permute_62);  primals_109 = permute_62 = None
        view_100: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_37, [1, 8, 768]);  addmm_37 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_101: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_100, [1, -1, 12, 64]);  view_100 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_63: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0])
        addmm_38: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_111, view_96, permute_64);  primals_111 = permute_64 = None
        view_103: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_38, [1, 8, 768]);  addmm_38 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_104: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_103, [1, -1, 12, 64]);  view_103 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_65: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_6 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_61, permute_63, permute_65, attn_mask = where)
        getitem_38: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_6[0]
        getitem_39: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_6[1];  _scaled_dot_product_flash_attention_for_cpu_6 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_66: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_105: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_66, [1, 8, 768]);  permute_66 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_106: "f32[8, 768]" = torch.ops.aten.view.default(view_105, [8, 768]);  view_105 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0])
        addmm_39: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_113, view_106, permute_67);  primals_113 = view_106 = permute_67 = None
        view_107: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_39, [1, 8, 768]);  addmm_39 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_46: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_107, add_45);  view_107 = add_45 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
        getitem_40: "f32[1, 8, 1]" = var_mean_13[0]
        getitem_41: "f32[1, 8, 1]" = var_mean_13[1];  var_mean_13 = None
        add_47: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_13: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
        sub_14: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_46, getitem_41);  add_46 = getitem_41 = None
        mul_44: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_13);  sub_14 = None
        mul_45: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_114)
        add_48: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_115);  mul_45 = primals_115 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_108: "f32[8, 768]" = torch.ops.aten.view.default(add_48, [8, 768])
        permute_68: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_116, [1, 0])
        addmm_40: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_117, view_108, permute_68);  primals_117 = permute_68 = None
        view_109: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_46: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_47: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_6: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_49: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_110: "f32[8, 3072]" = torch.ops.aten.view.default(mul_48, [8, 3072]);  mul_48 = None
        permute_69: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0])
        addmm_41: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_119, view_110, permute_69);  primals_119 = permute_69 = None
        view_111: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_41, [1, 8, 768]);  addmm_41 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_50: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_111, add_48);  view_111 = add_48 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
        getitem_42: "f32[1, 8, 1]" = var_mean_14[0]
        getitem_43: "f32[1, 8, 1]" = var_mean_14[1];  var_mean_14 = None
        add_51: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_14: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_15: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_50, getitem_43);  add_50 = getitem_43 = None
        mul_49: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_14);  sub_15 = None
        mul_50: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_120)
        add_52: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_121);  mul_50 = primals_121 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_112: "f32[8, 768]" = torch.ops.aten.view.default(add_52, [8, 768])
        permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0])
        addmm_42: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_123, view_112, permute_70);  primals_123 = permute_70 = None
        view_113: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_42, [1, 8, 768]);  addmm_42 = None
        view_114: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_113, [1, -1, 12, 64]);  view_113 = None
        permute_71: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_72: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0])
        addmm_43: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_125, view_112, permute_72);  primals_125 = permute_72 = None
        view_116: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_43, [1, 8, 768]);  addmm_43 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_117: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_116, [1, -1, 12, 64]);  view_116 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_73: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0])
        addmm_44: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_127, view_112, permute_74);  primals_127 = permute_74 = None
        view_119: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_44, [1, 8, 768]);  addmm_44 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_120: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_119, [1, -1, 12, 64]);  view_119 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_75: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_7 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_71, permute_73, permute_75, attn_mask = where)
        getitem_44: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_7[0]
        getitem_45: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_7[1];  _scaled_dot_product_flash_attention_for_cpu_7 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_76: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_44, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_121: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_76, [1, 8, 768]);  permute_76 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_122: "f32[8, 768]" = torch.ops.aten.view.default(view_121, [8, 768]);  view_121 = None
        permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0])
        addmm_45: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_129, view_122, permute_77);  primals_129 = view_122 = permute_77 = None
        view_123: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_45, [1, 8, 768]);  addmm_45 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_53: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_123, add_52);  view_123 = add_52 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_46: "f32[1, 8, 1]" = var_mean_15[0]
        getitem_47: "f32[1, 8, 1]" = var_mean_15[1];  var_mean_15 = None
        add_54: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_15: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_16: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_53, getitem_47);  add_53 = getitem_47 = None
        mul_51: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_15);  sub_16 = None
        mul_52: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_130)
        add_55: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_131);  mul_52 = primals_131 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_124: "f32[8, 768]" = torch.ops.aten.view.default(add_55, [8, 768])
        permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_132, [1, 0])
        addmm_46: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_133, view_124, permute_78);  primals_133 = permute_78 = None
        view_125: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_53: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.5)
        mul_54: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_125, 0.7071067811865476);  view_125 = None
        erf_7: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_56: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_126: "f32[8, 3072]" = torch.ops.aten.view.default(mul_55, [8, 3072]);  mul_55 = None
        permute_79: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_134, [1, 0])
        addmm_47: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_135, view_126, permute_79);  primals_135 = permute_79 = None
        view_127: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_47, [1, 8, 768]);  addmm_47 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_57: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_127, add_55);  view_127 = add_55 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_48: "f32[1, 8, 1]" = var_mean_16[0]
        getitem_49: "f32[1, 8, 1]" = var_mean_16[1];  var_mean_16 = None
        add_58: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_16: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_17: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_57, getitem_49);  add_57 = getitem_49 = None
        mul_56: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_16);  sub_17 = None
        mul_57: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_136)
        add_59: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_137);  mul_57 = primals_137 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_128: "f32[8, 768]" = torch.ops.aten.view.default(add_59, [8, 768])
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0])
        addmm_48: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_139, view_128, permute_80);  primals_139 = permute_80 = None
        view_129: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_48, [1, 8, 768]);  addmm_48 = None
        view_130: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_129, [1, -1, 12, 64]);  view_129 = None
        permute_81: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(primals_140, [1, 0])
        addmm_49: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_141, view_128, permute_82);  primals_141 = permute_82 = None
        view_132: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_49, [1, 8, 768]);  addmm_49 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_133: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_132, [1, -1, 12, 64]);  view_132 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_83: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_84: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0])
        addmm_50: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_143, view_128, permute_84);  primals_143 = permute_84 = None
        view_135: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_50, [1, 8, 768]);  addmm_50 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_136: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_135, [1, -1, 12, 64]);  view_135 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_85: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_8 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_81, permute_83, permute_85, attn_mask = where)
        getitem_50: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_8[0]
        getitem_51: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_8[1];  _scaled_dot_product_flash_attention_for_cpu_8 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_86: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_137: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_86, [1, 8, 768]);  permute_86 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_138: "f32[8, 768]" = torch.ops.aten.view.default(view_137, [8, 768]);  view_137 = None
        permute_87: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0])
        addmm_51: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_145, view_138, permute_87);  primals_145 = view_138 = permute_87 = None
        view_139: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_51, [1, 8, 768]);  addmm_51 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_60: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_139, add_59);  view_139 = add_59 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
        getitem_52: "f32[1, 8, 1]" = var_mean_17[0]
        getitem_53: "f32[1, 8, 1]" = var_mean_17[1];  var_mean_17 = None
        add_61: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_17: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_18: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_60, getitem_53);  add_60 = getitem_53 = None
        mul_58: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_17);  sub_18 = None
        mul_59: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_146)
        add_62: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_147);  mul_59 = primals_147 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_140: "f32[8, 768]" = torch.ops.aten.view.default(add_62, [8, 768])
        permute_88: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_148, [1, 0])
        addmm_52: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_149, view_140, permute_88);  primals_149 = permute_88 = None
        view_141: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_60: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.5)
        mul_61: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476);  view_141 = None
        erf_8: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_63: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_142: "f32[8, 3072]" = torch.ops.aten.view.default(mul_62, [8, 3072]);  mul_62 = None
        permute_89: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0])
        addmm_53: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_151, view_142, permute_89);  primals_151 = permute_89 = None
        view_143: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_53, [1, 8, 768]);  addmm_53 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_64: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_143, add_62);  view_143 = add_62 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
        getitem_54: "f32[1, 8, 1]" = var_mean_18[0]
        getitem_55: "f32[1, 8, 1]" = var_mean_18[1];  var_mean_18 = None
        add_65: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_18: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_19: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_64, getitem_55);  add_64 = getitem_55 = None
        mul_63: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_18);  sub_19 = None
        mul_64: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_152)
        add_66: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_153);  mul_64 = primals_153 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_144: "f32[8, 768]" = torch.ops.aten.view.default(add_66, [8, 768])
        permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0])
        addmm_54: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_155, view_144, permute_90);  primals_155 = permute_90 = None
        view_145: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_54, [1, 8, 768]);  addmm_54 = None
        view_146: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_145, [1, -1, 12, 64]);  view_145 = None
        permute_91: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0])
        addmm_55: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_157, view_144, permute_92);  primals_157 = permute_92 = None
        view_148: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_55, [1, 8, 768]);  addmm_55 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_149: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_148, [1, -1, 12, 64]);  view_148 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_93: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0])
        addmm_56: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_159, view_144, permute_94);  primals_159 = permute_94 = None
        view_151: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_56, [1, 8, 768]);  addmm_56 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_152: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_151, [1, -1, 12, 64]);  view_151 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_95: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_9 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_91, permute_93, permute_95, attn_mask = where)
        getitem_56: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_9[0]
        getitem_57: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_9[1];  _scaled_dot_product_flash_attention_for_cpu_9 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_96: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_56, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_153: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_96, [1, 8, 768]);  permute_96 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_154: "f32[8, 768]" = torch.ops.aten.view.default(view_153, [8, 768]);  view_153 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0])
        addmm_57: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_161, view_154, permute_97);  primals_161 = view_154 = permute_97 = None
        view_155: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_57, [1, 8, 768]);  addmm_57 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_67: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_155, add_66);  view_155 = add_66 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
        getitem_58: "f32[1, 8, 1]" = var_mean_19[0]
        getitem_59: "f32[1, 8, 1]" = var_mean_19[1];  var_mean_19 = None
        add_68: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_19: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        sub_20: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_59);  add_67 = getitem_59 = None
        mul_65: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_19);  sub_20 = None
        mul_66: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_162)
        add_69: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_163);  mul_66 = primals_163 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_156: "f32[8, 768]" = torch.ops.aten.view.default(add_69, [8, 768])
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_164, [1, 0])
        addmm_58: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_165, view_156, permute_98);  primals_165 = permute_98 = None
        view_157: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_67: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.5)
        mul_68: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_157, 0.7071067811865476);  view_157 = None
        erf_9: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_70: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_70);  mul_67 = add_70 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_158: "f32[8, 3072]" = torch.ops.aten.view.default(mul_69, [8, 3072]);  mul_69 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0])
        addmm_59: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_167, view_158, permute_99);  primals_167 = permute_99 = None
        view_159: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_59, [1, 8, 768]);  addmm_59 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_71: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_159, add_69);  view_159 = add_69 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
        getitem_60: "f32[1, 8, 1]" = var_mean_20[0]
        getitem_61: "f32[1, 8, 1]" = var_mean_20[1];  var_mean_20 = None
        add_72: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_20: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        sub_21: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_61);  add_71 = getitem_61 = None
        mul_70: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_20);  sub_21 = None
        mul_71: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_168)
        add_73: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_169);  mul_71 = primals_169 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_160: "f32[8, 768]" = torch.ops.aten.view.default(add_73, [8, 768])
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0])
        addmm_60: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_171, view_160, permute_100);  primals_171 = permute_100 = None
        view_161: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_60, [1, 8, 768]);  addmm_60 = None
        view_162: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_161, [1, -1, 12, 64]);  view_161 = None
        permute_101: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0])
        addmm_61: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_173, view_160, permute_102);  primals_173 = permute_102 = None
        view_164: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_61, [1, 8, 768]);  addmm_61 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_165: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_164, [1, -1, 12, 64]);  view_164 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_103: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0])
        addmm_62: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_175, view_160, permute_104);  primals_175 = permute_104 = None
        view_167: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_62, [1, 8, 768]);  addmm_62 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_168: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_167, [1, -1, 12, 64]);  view_167 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_105: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_10 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_101, permute_103, permute_105, attn_mask = where)
        getitem_62: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_10[0]
        getitem_63: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_10[1];  _scaled_dot_product_flash_attention_for_cpu_10 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_106: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_169: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_106, [1, 8, 768]);  permute_106 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_170: "f32[8, 768]" = torch.ops.aten.view.default(view_169, [8, 768]);  view_169 = None
        permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0])
        addmm_63: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_177, view_170, permute_107);  primals_177 = view_170 = permute_107 = None
        view_171: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_63, [1, 8, 768]);  addmm_63 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_74: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_171, add_73);  view_171 = add_73 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_64: "f32[1, 8, 1]" = var_mean_21[0]
        getitem_65: "f32[1, 8, 1]" = var_mean_21[1];  var_mean_21 = None
        add_75: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_21: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_22: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_74, getitem_65);  add_74 = getitem_65 = None
        mul_72: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_21);  sub_22 = None
        mul_73: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_178)
        add_76: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_179);  mul_73 = primals_179 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_172: "f32[8, 768]" = torch.ops.aten.view.default(add_76, [8, 768])
        permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_180, [1, 0])
        addmm_64: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_181, view_172, permute_108);  primals_181 = permute_108 = None
        view_173: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_74: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_75: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
        erf_10: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_77: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_77);  mul_74 = add_77 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_174: "f32[8, 3072]" = torch.ops.aten.view.default(mul_76, [8, 3072]);  mul_76 = None
        permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0])
        addmm_65: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_183, view_174, permute_109);  primals_183 = permute_109 = None
        view_175: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_65, [1, 8, 768]);  addmm_65 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_78: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_175, add_76);  view_175 = add_76 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
        getitem_66: "f32[1, 8, 1]" = var_mean_22[0]
        getitem_67: "f32[1, 8, 1]" = var_mean_22[1];  var_mean_22 = None
        add_79: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_22: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
        sub_23: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_78, getitem_67);  add_78 = getitem_67 = None
        mul_77: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_22);  sub_23 = None
        mul_78: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_184)
        add_80: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_185);  mul_78 = primals_185 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:363 in forward, code: self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        view_176: "f32[8, 768]" = torch.ops.aten.view.default(add_80, [8, 768])
        permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0])
        addmm_66: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_187, view_176, permute_110);  primals_187 = permute_110 = None
        view_177: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_66, [1, 8, 768]);  addmm_66 = None
        view_178: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_177, [1, -1, 12, 64]);  view_177 = None
        permute_111: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:387 in forward, code: self.key(current_states)
        permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0])
        addmm_67: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_189, view_176, permute_112);  primals_189 = permute_112 = None
        view_180: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_67, [1, 8, 768]);  addmm_67 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:388 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_181: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_180, [1, -1, 12, 64]);  view_180 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:389 in forward, code: .transpose(1, 2)
        permute_113: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:392 in forward, code: self.value(current_states)
        permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0])
        addmm_68: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_191, view_176, permute_114);  primals_191 = permute_114 = None
        view_183: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_68, [1, 8, 768]);  addmm_68 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:393 in forward, code: .view(bsz, -1, self.num_attention_heads, self.attention_head_size)
        view_184: "f32[1, 8, 12, 64]" = torch.ops.aten.view.default(view_183, [1, -1, 12, 64]);  view_183 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:394 in forward, code: .transpose(1, 2)
        permute_115: "f32[1, 12, 8, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:413 in forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_flash_attention_for_cpu_11 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(permute_111, permute_113, permute_115, attn_mask = where)
        getitem_68: "f32[1, 12, 8, 64]" = _scaled_dot_product_flash_attention_for_cpu_11[0]
        getitem_69: "f32[1, 12, 8]" = _scaled_dot_product_flash_attention_for_cpu_11[1];  _scaled_dot_product_flash_attention_for_cpu_11 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:422 in forward, code: attn_output = attn_output.transpose(1, 2)
        permute_116: "f32[1, 8, 12, 64]" = torch.ops.aten.permute.default(getitem_68, [0, 2, 1, 3])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:423 in forward, code: attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)
        view_185: "f32[1, 8, 768]" = torch.ops.aten.view.default(permute_116, [1, 8, 768]);  permute_116 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:436 in forward, code: hidden_states = self.dense(hidden_states)
        view_186: "f32[8, 768]" = torch.ops.aten.view.default(view_185, [8, 768]);  view_185 = None
        permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0])
        addmm_69: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_193, view_186, permute_117);  primals_193 = view_186 = permute_117 = None
        view_187: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_69, [1, 8, 768]);  addmm_69 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_81: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_187, add_80);  view_187 = add_80 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_70: "f32[1, 8, 1]" = var_mean_23[0]
        getitem_71: "f32[1, 8, 1]" = var_mean_23[1];  var_mean_23 = None
        add_82: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_23: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_24: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_81, getitem_71);  add_81 = getitem_71 = None
        mul_79: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_23);  sub_24 = None
        mul_80: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_194)
        add_83: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_195);  mul_80 = primals_195 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:512 in forward, code: hidden_states = self.dense(hidden_states)
        view_188: "f32[8, 768]" = torch.ops.aten.view.default(add_83, [8, 768])
        permute_118: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_196, [1, 0])
        addmm_70: "f32[8, 3072]" = torch.ops.aten.addmm.default(primals_197, view_188, permute_118);  primals_197 = permute_118 = None
        view_189: "f32[1, 8, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 8, 3072])
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/activations.py:85 in forward, code: return self.act(input)
        mul_81: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.5)
        mul_82: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(view_189, 0.7071067811865476);  view_189 = None
        erf_11: "f32[1, 8, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_84: "f32[1, 8, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[1, 8, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_84);  mul_81 = add_84 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:525 in forward, code: hidden_states = self.dense(hidden_states)
        view_190: "f32[8, 3072]" = torch.ops.aten.view.default(mul_83, [8, 3072]);  mul_83 = None
        permute_119: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0])
        addmm_71: "f32[8, 768]" = torch.ops.aten.addmm.default(primals_199, view_190, permute_119);  primals_199 = permute_119 = None
        view_191: "f32[1, 8, 768]" = torch.ops.aten.view.default(addmm_71, [1, 8, 768]);  addmm_71 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        add_85: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(view_191, add_83);  view_191 = add_83 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_72: "f32[1, 8, 1]" = var_mean_24[0]
        getitem_73: "f32[1, 8, 1]" = var_mean_24[1];  var_mean_24 = None
        add_86: "f32[1, 8, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_24: "f32[1, 8, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_25: "f32[1, 8, 768]" = torch.ops.aten.sub.Tensor(add_85, getitem_73);  add_85 = getitem_73 = None
        mul_84: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
        mul_85: "f32[1, 8, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_200)
        add_87: "f32[1, 8, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_201);  mul_85 = primals_201 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:700 in forward, code: first_token_tensor = hidden_states[:, 0]
        select: "f32[1, 768]" = torch.ops.aten.select.int(add_87, 1, 0)
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:701 in forward, code: pooled_output = self.dense(first_token_tensor)
        permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(primals_202, [1, 0])
        addmm_72: "f32[1, 768]" = torch.ops.aten.addmm.default(primals_203, select, permute_120);  primals_203 = permute_120 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:702 in forward, code: pooled_output = self.activation(pooled_output)
        tanh: "f32[1, 768]" = torch.ops.aten.tanh.default(addmm_72);  addmm_72 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_1: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_2: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_3: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_4: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_5: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_6: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_7: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_8: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_9: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_10: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_11: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_12: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_13: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_14: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_15: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_16: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_17: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_18: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_19: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_20: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_21: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:527 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_22: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:438 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        div_23: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        
        # File: /opt/pytorch/lib/python3.12/site-packages/transformers/models/bert/modeling_bert.py:186 in forward, code: embeddings = self.LayerNorm(embeddings)
        div_24: "f32[1, 8, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        return (add_87, tanh, primals_1, primals_2, primals_3, primals_7, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, mul, where, view, permute_1, permute_3, permute_5, getitem_2, getitem_3, mul_2, view_12, addmm_4, view_14, mul_7, view_16, permute_11, permute_13, permute_15, getitem_8, getitem_9, mul_9, view_28, addmm_10, view_30, mul_14, view_32, permute_21, permute_23, permute_25, getitem_14, getitem_15, mul_16, view_44, addmm_16, view_46, mul_21, view_48, permute_31, permute_33, permute_35, getitem_20, getitem_21, mul_23, view_60, addmm_22, view_62, mul_28, view_64, permute_41, permute_43, permute_45, getitem_26, getitem_27, mul_30, view_76, addmm_28, view_78, mul_35, view_80, permute_51, permute_53, permute_55, getitem_32, getitem_33, mul_37, view_92, addmm_34, view_94, mul_42, view_96, permute_61, permute_63, permute_65, getitem_38, getitem_39, mul_44, view_108, addmm_40, view_110, mul_49, view_112, permute_71, permute_73, permute_75, getitem_44, getitem_45, mul_51, view_124, addmm_46, view_126, mul_56, view_128, permute_81, permute_83, permute_85, getitem_50, getitem_51, mul_58, view_140, addmm_52, view_142, mul_63, view_144, permute_91, permute_93, permute_95, getitem_56, getitem_57, mul_65, view_156, addmm_58, view_158, mul_70, view_160, permute_101, permute_103, permute_105, getitem_62, getitem_63, mul_72, view_172, addmm_64, view_174, mul_77, view_176, permute_111, permute_113, permute_115, getitem_68, getitem_69, mul_79, view_188, addmm_70, view_190, mul_84, select, tanh, div, div_1, div_2, div_3, div_4, div_5, div_6, div_7, div_8, div_9, div_10, div_11, div_12, div_13, div_14, div_15, div_16, div_17, div_18, div_19, div_20, div_21, div_22, div_23, div_24)
        