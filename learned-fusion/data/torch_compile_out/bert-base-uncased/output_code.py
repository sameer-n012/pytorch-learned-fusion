# AOT ID: ['0_forward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


cpp_fused__to_copy_add_embedding_expand_lift_fresh_masked_fill_native_layer_norm_native_layer_norm_backward_slice_sub_unsqueeze_0 = async_compile.cpp_pybinding(['float*', 'const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const int64_t*', 'const float*', 'const float*', 'const float*', 'const int64_t*', 'float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const int64_t* in_ptr0,
                       const float* in_ptr1,
                       const int64_t* in_ptr2,
                       const float* in_ptr3,
                       const int64_t* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const int64_t* in_ptr8,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                            auto tmp10 = in_ptr2[static_cast<int64_t>(x0)];
                            auto tmp21 = in_ptr4[static_cast<int64_t>(x0)];
                            auto tmp1 = 30522L;
                            auto tmp2 = c10::convert<int64_t>(tmp1);
                            auto tmp3 = int64_t(tmp0 + tmp2);
                            auto tmp4 = tmp0 < 0;
                            auto tmp5 = tmp4 ? tmp3 : tmp0;
                            auto tmp6 = tmp5;
                            auto tmp7 = c10::convert<int64_t>(tmp6);
                            TORCH_CHECK((0 <= tmp7) & (tmp7 < 30522L), "index out of bounds: 0 <= tmp7 < 30522L");
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*tmp5), static_cast<int64_t>(8));
                            auto tmp11 = 2L;
                            auto tmp12 = c10::convert<int64_t>(tmp11);
                            auto tmp13 = int64_t(tmp10 + tmp12);
                            auto tmp14 = tmp10 < 0;
                            auto tmp15 = tmp14 ? tmp13 : tmp10;
                            auto tmp16 = tmp15;
                            auto tmp17 = c10::convert<int64_t>(tmp16);
                            TORCH_CHECK((0 <= tmp17) & (tmp17 < 2L), "index out of bounds: 0 <= tmp17 < 2L");
                            auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1 + 768L*tmp15), static_cast<int64_t>(8));
                            auto tmp20 = tmp9 + tmp19;
                            auto tmp22 = 512L;
                            auto tmp23 = c10::convert<int64_t>(tmp22);
                            auto tmp24 = int64_t(tmp21 + tmp23);
                            auto tmp25 = tmp21 < 0;
                            auto tmp26 = tmp25 ? tmp24 : tmp21;
                            auto tmp27 = tmp26;
                            auto tmp28 = c10::convert<int64_t>(tmp27);
                            TORCH_CHECK((0 <= tmp28) & (tmp28 < 512L), "index out of bounds: 0 <= tmp28 < 512L");
                            auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<int64_t>(x1 + 768L*tmp26), static_cast<int64_t>(8));
                            auto tmp31 = tmp20 + tmp30;
                            tmp31.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp31, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr2[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp4 = out_ptr2[static_cast<int64_t>(x0)];
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 - tmp2;
                        auto tmp5 = static_cast<float>(768.0);
                        auto tmp6 = tmp4 / tmp5;
                        auto tmp7 = static_cast<float>(1e-12);
                        auto tmp8 = float(tmp6 + tmp7);
                        auto tmp9 = 1 / std::sqrt(tmp8);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp3 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp11.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp15.store(out_ptr3 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr4 + static_cast<int64_t>(x0));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(8L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(8L)))
                    {
                        auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr8 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::convert<float,1,int64_t,2>(tmp0);
                        auto tmp2 = static_cast<float>(1.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp3 - tmp1;
                        auto tmp5 = at::vec::VecMask<float,1>::from<float,1>(tmp4);
                        auto tmp6 = static_cast<float>(-3.4028234663852886e+38);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = decltype(tmp7)::blendv(tmp4, tmp7, tmp5.template cast<float,1>());
                        tmp8.store(out_ptr5 + static_cast<int64_t>(x1 + 8L*x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_1 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_2 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_3 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_4 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_5 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_6 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_7 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_8 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_9 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_10 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_11 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_12 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_13 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_14 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_15 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_16 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_17 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_18 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_19 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_20 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_21 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_22 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_23 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_24 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_25 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_26 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_27 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_28 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_29 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_30 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_31 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_32 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_33 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_34 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp_pybinding(['const float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(24)
    {
        int tid = omp_get_thread_num();
        {
            #pragma omp for
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(24576L); x0+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(24576L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                        auto tmp1 = static_cast<float>(0.5);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp4 = static_cast<float>(0.7071067811865476);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp0 * tmp5;
                        auto tmp7 = tmp6.erf();
                        auto tmp8 = static_cast<float>(1.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp3 * tmp10;
                        tmp11.store(out_ptr0 + static_cast<int64_t>(x0));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_36 = async_compile.cpp_pybinding(['float*', 'const float*', 'const float*', 'const float*', 'float*', 'float*', 'float*', 'float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<at::vec::Vectorized<float>> masked_tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                static WelfordHelper<at::vec::Vectorized<float>, 4096> welford_helper0(static_cast<int64_t>(96L));
                static WelfordHelper<at::vec::Vectorized<float>, 4096> masked_welford_helper0(static_cast<int64_t>(0L));
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2, &welford_helper0);
                        }
                    }
                }
                tmp_acc0_vec = welford_combine(tmp_acc0_vec, &welford_helper0);
                masked_tmp_acc0_vec = welford_combine(masked_tmp_acc0_vec, &masked_welford_helper0);
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(masked_tmp_acc0_vec));
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.mean);
                out_ptr1[static_cast<int64_t>(x0)] = static_cast<float>(tmp_acc0.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(768L); x1+=static_cast<int64_t>(8L))
            {
                {
                    if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(768L)))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x1 + 768L*x0), static_cast<int64_t>(8));
                        auto tmp3 = out_ptr0[static_cast<int64_t>(x0)];
                        auto tmp6 = out_ptr1[static_cast<int64_t>(x0)];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<int64_t>(x1), static_cast<int64_t>(8));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-12);
                        auto tmp10 = float(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp13.store(in_out_ptr0 + static_cast<int64_t>(x1 + 768L*x0));
                        tmp17.store(out_ptr2 + static_cast<int64_t>(x1 + 768L*x0));
                    }
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-12);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(out_ptr3 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_tanh_37 = async_compile.cpp_pybinding(['float*'], '''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(float* in_out_ptr0)
{
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(768L); x0+=static_cast<int64_t>(8L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(768L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8));
                    auto tmp1 = tmp0.tanh();
                    tmp1.store(in_out_ptr0 + static_cast<int64_t>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203 = args
        args.clear()
        assert_size_stride(primals_1, (1, 8), (8, 1))
        assert_size_stride(primals_2, (1, 512), (512, 1))
        assert_size_stride(primals_3, (1, 512), (512, 1))
        assert_size_stride(primals_4, (30522, 768), (768, 1))
        assert_size_stride(primals_5, (2, 768), (768, 1))
        assert_size_stride(primals_6, (512, 768), (768, 1))
        assert_size_stride(primals_7, (768, ), (1, ))
        assert_size_stride(primals_8, (768, ), (1, ))
        assert_size_stride(primals_9, (1, 8), (8, 1))
        assert_size_stride(primals_10, (768, 768), (768, 1))
        assert_size_stride(primals_11, (768, ), (1, ))
        assert_size_stride(primals_12, (768, 768), (768, 1))
        assert_size_stride(primals_13, (768, ), (1, ))
        assert_size_stride(primals_14, (768, 768), (768, 1))
        assert_size_stride(primals_15, (768, ), (1, ))
        assert_size_stride(primals_16, (768, 768), (768, 1))
        assert_size_stride(primals_17, (768, ), (1, ))
        assert_size_stride(primals_18, (768, ), (1, ))
        assert_size_stride(primals_19, (768, ), (1, ))
        assert_size_stride(primals_20, (3072, 768), (768, 1))
        assert_size_stride(primals_21, (3072, ), (1, ))
        assert_size_stride(primals_22, (768, 3072), (3072, 1))
        assert_size_stride(primals_23, (768, ), (1, ))
        assert_size_stride(primals_24, (768, ), (1, ))
        assert_size_stride(primals_25, (768, ), (1, ))
        assert_size_stride(primals_26, (768, 768), (768, 1))
        assert_size_stride(primals_27, (768, ), (1, ))
        assert_size_stride(primals_28, (768, 768), (768, 1))
        assert_size_stride(primals_29, (768, ), (1, ))
        assert_size_stride(primals_30, (768, 768), (768, 1))
        assert_size_stride(primals_31, (768, ), (1, ))
        assert_size_stride(primals_32, (768, 768), (768, 1))
        assert_size_stride(primals_33, (768, ), (1, ))
        assert_size_stride(primals_34, (768, ), (1, ))
        assert_size_stride(primals_35, (768, ), (1, ))
        assert_size_stride(primals_36, (3072, 768), (768, 1))
        assert_size_stride(primals_37, (3072, ), (1, ))
        assert_size_stride(primals_38, (768, 3072), (3072, 1))
        assert_size_stride(primals_39, (768, ), (1, ))
        assert_size_stride(primals_40, (768, ), (1, ))
        assert_size_stride(primals_41, (768, ), (1, ))
        assert_size_stride(primals_42, (768, 768), (768, 1))
        assert_size_stride(primals_43, (768, ), (1, ))
        assert_size_stride(primals_44, (768, 768), (768, 1))
        assert_size_stride(primals_45, (768, ), (1, ))
        assert_size_stride(primals_46, (768, 768), (768, 1))
        assert_size_stride(primals_47, (768, ), (1, ))
        assert_size_stride(primals_48, (768, 768), (768, 1))
        assert_size_stride(primals_49, (768, ), (1, ))
        assert_size_stride(primals_50, (768, ), (1, ))
        assert_size_stride(primals_51, (768, ), (1, ))
        assert_size_stride(primals_52, (3072, 768), (768, 1))
        assert_size_stride(primals_53, (3072, ), (1, ))
        assert_size_stride(primals_54, (768, 3072), (3072, 1))
        assert_size_stride(primals_55, (768, ), (1, ))
        assert_size_stride(primals_56, (768, ), (1, ))
        assert_size_stride(primals_57, (768, ), (1, ))
        assert_size_stride(primals_58, (768, 768), (768, 1))
        assert_size_stride(primals_59, (768, ), (1, ))
        assert_size_stride(primals_60, (768, 768), (768, 1))
        assert_size_stride(primals_61, (768, ), (1, ))
        assert_size_stride(primals_62, (768, 768), (768, 1))
        assert_size_stride(primals_63, (768, ), (1, ))
        assert_size_stride(primals_64, (768, 768), (768, 1))
        assert_size_stride(primals_65, (768, ), (1, ))
        assert_size_stride(primals_66, (768, ), (1, ))
        assert_size_stride(primals_67, (768, ), (1, ))
        assert_size_stride(primals_68, (3072, 768), (768, 1))
        assert_size_stride(primals_69, (3072, ), (1, ))
        assert_size_stride(primals_70, (768, 3072), (3072, 1))
        assert_size_stride(primals_71, (768, ), (1, ))
        assert_size_stride(primals_72, (768, ), (1, ))
        assert_size_stride(primals_73, (768, ), (1, ))
        assert_size_stride(primals_74, (768, 768), (768, 1))
        assert_size_stride(primals_75, (768, ), (1, ))
        assert_size_stride(primals_76, (768, 768), (768, 1))
        assert_size_stride(primals_77, (768, ), (1, ))
        assert_size_stride(primals_78, (768, 768), (768, 1))
        assert_size_stride(primals_79, (768, ), (1, ))
        assert_size_stride(primals_80, (768, 768), (768, 1))
        assert_size_stride(primals_81, (768, ), (1, ))
        assert_size_stride(primals_82, (768, ), (1, ))
        assert_size_stride(primals_83, (768, ), (1, ))
        assert_size_stride(primals_84, (3072, 768), (768, 1))
        assert_size_stride(primals_85, (3072, ), (1, ))
        assert_size_stride(primals_86, (768, 3072), (3072, 1))
        assert_size_stride(primals_87, (768, ), (1, ))
        assert_size_stride(primals_88, (768, ), (1, ))
        assert_size_stride(primals_89, (768, ), (1, ))
        assert_size_stride(primals_90, (768, 768), (768, 1))
        assert_size_stride(primals_91, (768, ), (1, ))
        assert_size_stride(primals_92, (768, 768), (768, 1))
        assert_size_stride(primals_93, (768, ), (1, ))
        assert_size_stride(primals_94, (768, 768), (768, 1))
        assert_size_stride(primals_95, (768, ), (1, ))
        assert_size_stride(primals_96, (768, 768), (768, 1))
        assert_size_stride(primals_97, (768, ), (1, ))
        assert_size_stride(primals_98, (768, ), (1, ))
        assert_size_stride(primals_99, (768, ), (1, ))
        assert_size_stride(primals_100, (3072, 768), (768, 1))
        assert_size_stride(primals_101, (3072, ), (1, ))
        assert_size_stride(primals_102, (768, 3072), (3072, 1))
        assert_size_stride(primals_103, (768, ), (1, ))
        assert_size_stride(primals_104, (768, ), (1, ))
        assert_size_stride(primals_105, (768, ), (1, ))
        assert_size_stride(primals_106, (768, 768), (768, 1))
        assert_size_stride(primals_107, (768, ), (1, ))
        assert_size_stride(primals_108, (768, 768), (768, 1))
        assert_size_stride(primals_109, (768, ), (1, ))
        assert_size_stride(primals_110, (768, 768), (768, 1))
        assert_size_stride(primals_111, (768, ), (1, ))
        assert_size_stride(primals_112, (768, 768), (768, 1))
        assert_size_stride(primals_113, (768, ), (1, ))
        assert_size_stride(primals_114, (768, ), (1, ))
        assert_size_stride(primals_115, (768, ), (1, ))
        assert_size_stride(primals_116, (3072, 768), (768, 1))
        assert_size_stride(primals_117, (3072, ), (1, ))
        assert_size_stride(primals_118, (768, 3072), (3072, 1))
        assert_size_stride(primals_119, (768, ), (1, ))
        assert_size_stride(primals_120, (768, ), (1, ))
        assert_size_stride(primals_121, (768, ), (1, ))
        assert_size_stride(primals_122, (768, 768), (768, 1))
        assert_size_stride(primals_123, (768, ), (1, ))
        assert_size_stride(primals_124, (768, 768), (768, 1))
        assert_size_stride(primals_125, (768, ), (1, ))
        assert_size_stride(primals_126, (768, 768), (768, 1))
        assert_size_stride(primals_127, (768, ), (1, ))
        assert_size_stride(primals_128, (768, 768), (768, 1))
        assert_size_stride(primals_129, (768, ), (1, ))
        assert_size_stride(primals_130, (768, ), (1, ))
        assert_size_stride(primals_131, (768, ), (1, ))
        assert_size_stride(primals_132, (3072, 768), (768, 1))
        assert_size_stride(primals_133, (3072, ), (1, ))
        assert_size_stride(primals_134, (768, 3072), (3072, 1))
        assert_size_stride(primals_135, (768, ), (1, ))
        assert_size_stride(primals_136, (768, ), (1, ))
        assert_size_stride(primals_137, (768, ), (1, ))
        assert_size_stride(primals_138, (768, 768), (768, 1))
        assert_size_stride(primals_139, (768, ), (1, ))
        assert_size_stride(primals_140, (768, 768), (768, 1))
        assert_size_stride(primals_141, (768, ), (1, ))
        assert_size_stride(primals_142, (768, 768), (768, 1))
        assert_size_stride(primals_143, (768, ), (1, ))
        assert_size_stride(primals_144, (768, 768), (768, 1))
        assert_size_stride(primals_145, (768, ), (1, ))
        assert_size_stride(primals_146, (768, ), (1, ))
        assert_size_stride(primals_147, (768, ), (1, ))
        assert_size_stride(primals_148, (3072, 768), (768, 1))
        assert_size_stride(primals_149, (3072, ), (1, ))
        assert_size_stride(primals_150, (768, 3072), (3072, 1))
        assert_size_stride(primals_151, (768, ), (1, ))
        assert_size_stride(primals_152, (768, ), (1, ))
        assert_size_stride(primals_153, (768, ), (1, ))
        assert_size_stride(primals_154, (768, 768), (768, 1))
        assert_size_stride(primals_155, (768, ), (1, ))
        assert_size_stride(primals_156, (768, 768), (768, 1))
        assert_size_stride(primals_157, (768, ), (1, ))
        assert_size_stride(primals_158, (768, 768), (768, 1))
        assert_size_stride(primals_159, (768, ), (1, ))
        assert_size_stride(primals_160, (768, 768), (768, 1))
        assert_size_stride(primals_161, (768, ), (1, ))
        assert_size_stride(primals_162, (768, ), (1, ))
        assert_size_stride(primals_163, (768, ), (1, ))
        assert_size_stride(primals_164, (3072, 768), (768, 1))
        assert_size_stride(primals_165, (3072, ), (1, ))
        assert_size_stride(primals_166, (768, 3072), (3072, 1))
        assert_size_stride(primals_167, (768, ), (1, ))
        assert_size_stride(primals_168, (768, ), (1, ))
        assert_size_stride(primals_169, (768, ), (1, ))
        assert_size_stride(primals_170, (768, 768), (768, 1))
        assert_size_stride(primals_171, (768, ), (1, ))
        assert_size_stride(primals_172, (768, 768), (768, 1))
        assert_size_stride(primals_173, (768, ), (1, ))
        assert_size_stride(primals_174, (768, 768), (768, 1))
        assert_size_stride(primals_175, (768, ), (1, ))
        assert_size_stride(primals_176, (768, 768), (768, 1))
        assert_size_stride(primals_177, (768, ), (1, ))
        assert_size_stride(primals_178, (768, ), (1, ))
        assert_size_stride(primals_179, (768, ), (1, ))
        assert_size_stride(primals_180, (3072, 768), (768, 1))
        assert_size_stride(primals_181, (3072, ), (1, ))
        assert_size_stride(primals_182, (768, 3072), (3072, 1))
        assert_size_stride(primals_183, (768, ), (1, ))
        assert_size_stride(primals_184, (768, ), (1, ))
        assert_size_stride(primals_185, (768, ), (1, ))
        assert_size_stride(primals_186, (768, 768), (768, 1))
        assert_size_stride(primals_187, (768, ), (1, ))
        assert_size_stride(primals_188, (768, 768), (768, 1))
        assert_size_stride(primals_189, (768, ), (1, ))
        assert_size_stride(primals_190, (768, 768), (768, 1))
        assert_size_stride(primals_191, (768, ), (1, ))
        assert_size_stride(primals_192, (768, 768), (768, 1))
        assert_size_stride(primals_193, (768, ), (1, ))
        assert_size_stride(primals_194, (768, ), (1, ))
        assert_size_stride(primals_195, (768, ), (1, ))
        assert_size_stride(primals_196, (3072, 768), (768, 1))
        assert_size_stride(primals_197, (3072, ), (1, ))
        assert_size_stride(primals_198, (768, 3072), (3072, 1))
        assert_size_stride(primals_199, (768, ), (1, ))
        assert_size_stride(primals_200, (768, ), (1, ))
        assert_size_stride(primals_201, (768, ), (1, ))
        assert_size_stride(primals_202, (768, 768), (768, 1))
        assert_size_stride(primals_203, (768, ), (1, ))
        buf0 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf1 = empty_strided_cpu((1, 8, 1), (8, 1, 8), torch.float32)
        buf2 = empty_strided_cpu((1, 8, 1), (8, 1, 8), torch.float32)
        buf4 = buf0; del buf0  # reuse
        buf6 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf273 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        buf5 = empty_strided_cpu((1, 1, 8, 8), (64, 64, 8, 1), torch.float32)
        cpp_fused__to_copy_add_embedding_expand_lift_fresh_masked_fill_native_layer_norm_native_layer_norm_backward_slice_sub_unsqueeze_0(buf4, primals_1, primals_4, primals_2, primals_5, primals_3, primals_6, primals_7, primals_8, primals_9, buf1, buf2, buf6, buf273, buf5)
        del primals_4
        del primals_5
        del primals_6
        del primals_8
        del primals_9
        buf7 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embeddings_2, linear], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_11, reinterpret_tensor(buf6, (8, 768), (768, 1), 0), reinterpret_tensor(primals_10, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
        del primals_11
        buf8 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embeddings_2, linear, linear_1], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_13, reinterpret_tensor(buf6, (8, 768), (768, 1), 0), reinterpret_tensor(primals_12, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
        del primals_13
        buf9 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embeddings_2, linear, linear_2], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_15, reinterpret_tensor(buf6, (8, 768), (768, 1), 0), reinterpret_tensor(primals_14, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
        del primals_15
        # Topologically Sorted Source Nodes: [linear, view, query_layer, linear_1, view_1, key_layer, linear_2, view_2, value_layer, attn_output], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf10 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf7, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf9, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf11 = buf10[0]
        assert_size_stride(buf11, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf11, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf12 = buf10[1]
        assert_size_stride(buf12, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf12, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf10
        buf13 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_1, attn_output_2, hidden_states], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_17, reinterpret_tensor(buf11, (8, 768), (768, 1), 0), reinterpret_tensor(primals_16, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf13)
        del primals_17
        buf14 = buf2; del buf2  # reuse
        buf15 = buf1; del buf1  # reuse
        buf17 = reinterpret_tensor(buf13, (1, 8, 768), (6144, 768, 1), 0); del buf13  # reuse
        buf18 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf272 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_1(buf17, buf6, primals_18, primals_19, buf14, buf15, buf18, buf272)
        del primals_19
        buf19 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_2, hidden_states_3], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf18, (8, 768), (768, 1), 0), reinterpret_tensor(primals_20, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf19)
        del primals_21
        buf20 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_2(buf19, buf20)
        buf21 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_3, hidden_states_4, hidden_states_5], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_23, reinterpret_tensor(buf20, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_22, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf21)
        del primals_23
        buf22 = buf15; del buf15  # reuse
        buf23 = buf14; del buf14  # reuse
        buf25 = reinterpret_tensor(buf21, (1, 8, 768), (6144, 768, 1), 0); del buf21  # reuse
        buf26 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf271 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_3(buf25, buf18, primals_24, primals_25, buf22, buf23, buf26, buf271)
        del primals_25
        buf27 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_7, linear_6], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf26, (8, 768), (768, 1), 0), reinterpret_tensor(primals_26, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
        del primals_27
        buf28 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_7, linear_6, linear_7], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_29, reinterpret_tensor(buf26, (8, 768), (768, 1), 0), reinterpret_tensor(primals_28, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf28)
        del primals_29
        buf29 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_7, linear_6, linear_8], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_31, reinterpret_tensor(buf26, (8, 768), (768, 1), 0), reinterpret_tensor(primals_30, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf29)
        del primals_31
        # Topologically Sorted Source Nodes: [linear_6, view_3, query_layer_1, linear_7, view_4, key_layer_1, linear_8, view_5, value_layer_1, attn_output_3], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf30 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf27, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf28, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf29, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf31 = buf30[0]
        assert_size_stride(buf31, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf31, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf32 = buf30[1]
        assert_size_stride(buf32, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf32, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf30
        buf33 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_4, attn_output_5, hidden_states_8], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_33, reinterpret_tensor(buf31, (8, 768), (768, 1), 0), reinterpret_tensor(primals_32, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf33)
        del primals_33
        buf34 = buf23; del buf23  # reuse
        buf35 = buf22; del buf22  # reuse
        buf37 = reinterpret_tensor(buf33, (1, 8, 768), (6144, 768, 1), 0); del buf33  # reuse
        buf38 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf270 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_4(buf37, buf26, primals_34, primals_35, buf34, buf35, buf38, buf270)
        del primals_35
        buf39 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_10, hidden_states_11], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_37, reinterpret_tensor(buf38, (8, 768), (768, 1), 0), reinterpret_tensor(primals_36, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf39)
        del primals_37
        buf40 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_5(buf39, buf40)
        buf41 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_11, hidden_states_12, hidden_states_13], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_39, reinterpret_tensor(buf40, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_38, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf41)
        del primals_39
        buf42 = buf35; del buf35  # reuse
        buf43 = buf34; del buf34  # reuse
        buf45 = reinterpret_tensor(buf41, (1, 8, 768), (6144, 768, 1), 0); del buf41  # reuse
        buf46 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf269 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_6(buf45, buf38, primals_40, primals_41, buf42, buf43, buf46, buf269)
        del primals_41
        buf47 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_15, linear_12], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf46, (8, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
        del primals_43
        buf48 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_15, linear_12, linear_13], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_45, reinterpret_tensor(buf46, (8, 768), (768, 1), 0), reinterpret_tensor(primals_44, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf48)
        del primals_45
        buf49 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_15, linear_12, linear_14], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_47, reinterpret_tensor(buf46, (8, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf49)
        del primals_47
        # Topologically Sorted Source Nodes: [linear_12, view_6, query_layer_2, linear_13, view_7, key_layer_2, linear_14, view_8, value_layer_2, attn_output_6], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf50 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf47, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf48, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf49, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf51 = buf50[0]
        assert_size_stride(buf51, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf51, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf52 = buf50[1]
        assert_size_stride(buf52, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf52, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf50
        buf53 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_7, attn_output_8, hidden_states_16], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_49, reinterpret_tensor(buf51, (8, 768), (768, 1), 0), reinterpret_tensor(primals_48, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
        del primals_49
        buf54 = buf43; del buf43  # reuse
        buf55 = buf42; del buf42  # reuse
        buf57 = reinterpret_tensor(buf53, (1, 8, 768), (6144, 768, 1), 0); del buf53  # reuse
        buf58 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf268 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_7(buf57, buf46, primals_50, primals_51, buf54, buf55, buf58, buf268)
        del primals_51
        buf59 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_18, hidden_states_19], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_53, reinterpret_tensor(buf58, (8, 768), (768, 1), 0), reinterpret_tensor(primals_52, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf59)
        del primals_53
        buf60 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_8(buf59, buf60)
        buf61 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_19, hidden_states_20, hidden_states_21], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_55, reinterpret_tensor(buf60, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_54, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf61)
        del primals_55
        buf62 = buf55; del buf55  # reuse
        buf63 = buf54; del buf54  # reuse
        buf65 = reinterpret_tensor(buf61, (1, 8, 768), (6144, 768, 1), 0); del buf61  # reuse
        buf66 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf267 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_9(buf65, buf58, primals_56, primals_57, buf62, buf63, buf66, buf267)
        del primals_57
        buf67 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_23, linear_18], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_59, reinterpret_tensor(buf66, (8, 768), (768, 1), 0), reinterpret_tensor(primals_58, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf67)
        del primals_59
        buf68 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_23, linear_18, linear_19], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_61, reinterpret_tensor(buf66, (8, 768), (768, 1), 0), reinterpret_tensor(primals_60, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf68)
        del primals_61
        buf69 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_23, linear_18, linear_20], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_63, reinterpret_tensor(buf66, (8, 768), (768, 1), 0), reinterpret_tensor(primals_62, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf69)
        del primals_63
        # Topologically Sorted Source Nodes: [linear_18, view_9, query_layer_3, linear_19, view_10, key_layer_3, linear_20, view_11, value_layer_3, attn_output_9], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf70 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf67, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf68, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf69, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf71 = buf70[0]
        assert_size_stride(buf71, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf71, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf72 = buf70[1]
        assert_size_stride(buf72, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf72, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf70
        buf73 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_10, attn_output_11, hidden_states_24], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_65, reinterpret_tensor(buf71, (8, 768), (768, 1), 0), reinterpret_tensor(primals_64, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
        del primals_65
        buf74 = buf63; del buf63  # reuse
        buf75 = buf62; del buf62  # reuse
        buf77 = reinterpret_tensor(buf73, (1, 8, 768), (6144, 768, 1), 0); del buf73  # reuse
        buf78 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf266 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_10(buf77, buf66, primals_66, primals_67, buf74, buf75, buf78, buf266)
        del primals_67
        buf79 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_26, hidden_states_27], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_69, reinterpret_tensor(buf78, (8, 768), (768, 1), 0), reinterpret_tensor(primals_68, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf79)
        del primals_69
        buf80 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_11(buf79, buf80)
        buf81 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_27, hidden_states_28, hidden_states_29], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_71, reinterpret_tensor(buf80, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_70, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf81)
        del primals_71
        buf82 = buf75; del buf75  # reuse
        buf83 = buf74; del buf74  # reuse
        buf85 = reinterpret_tensor(buf81, (1, 8, 768), (6144, 768, 1), 0); del buf81  # reuse
        buf86 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf265 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_12(buf85, buf78, primals_72, primals_73, buf82, buf83, buf86, buf265)
        del primals_73
        buf87 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_31, linear_24], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_75, reinterpret_tensor(buf86, (8, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf87)
        del primals_75
        buf88 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_31, linear_24, linear_25], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_77, reinterpret_tensor(buf86, (8, 768), (768, 1), 0), reinterpret_tensor(primals_76, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf88)
        del primals_77
        buf89 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_31, linear_24, linear_26], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_79, reinterpret_tensor(buf86, (8, 768), (768, 1), 0), reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
        del primals_79
        # Topologically Sorted Source Nodes: [linear_24, view_12, query_layer_4, linear_25, view_13, key_layer_4, linear_26, view_14, value_layer_4, attn_output_12], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf90 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf87, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf88, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf89, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf91 = buf90[0]
        assert_size_stride(buf91, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf91, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf92 = buf90[1]
        assert_size_stride(buf92, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf92, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf90
        buf93 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_13, attn_output_14, hidden_states_32], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_81, reinterpret_tensor(buf91, (8, 768), (768, 1), 0), reinterpret_tensor(primals_80, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf93)
        del primals_81
        buf94 = buf83; del buf83  # reuse
        buf95 = buf82; del buf82  # reuse
        buf97 = reinterpret_tensor(buf93, (1, 8, 768), (6144, 768, 1), 0); del buf93  # reuse
        buf98 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf264 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_13(buf97, buf86, primals_82, primals_83, buf94, buf95, buf98, buf264)
        del primals_83
        buf99 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_85, reinterpret_tensor(buf98, (8, 768), (768, 1), 0), reinterpret_tensor(primals_84, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf99)
        del primals_85
        buf100 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_14(buf99, buf100)
        buf101 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_35, hidden_states_36, hidden_states_37], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_87, reinterpret_tensor(buf100, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_86, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf101)
        del primals_87
        buf102 = buf95; del buf95  # reuse
        buf103 = buf94; del buf94  # reuse
        buf105 = reinterpret_tensor(buf101, (1, 8, 768), (6144, 768, 1), 0); del buf101  # reuse
        buf106 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf263 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_15(buf105, buf98, primals_88, primals_89, buf102, buf103, buf106, buf263)
        del primals_89
        buf107 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_39, linear_30], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf106, (8, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf107)
        del primals_91
        buf108 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_39, linear_30, linear_31], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_93, reinterpret_tensor(buf106, (8, 768), (768, 1), 0), reinterpret_tensor(primals_92, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf108)
        del primals_93
        buf109 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_39, linear_30, linear_32], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_95, reinterpret_tensor(buf106, (8, 768), (768, 1), 0), reinterpret_tensor(primals_94, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf109)
        del primals_95
        # Topologically Sorted Source Nodes: [linear_30, view_15, query_layer_5, linear_31, view_16, key_layer_5, linear_32, view_17, value_layer_5, attn_output_15], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf110 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf107, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf108, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf109, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf111 = buf110[0]
        assert_size_stride(buf111, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf111, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf112 = buf110[1]
        assert_size_stride(buf112, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf112, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf110
        buf113 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_16, attn_output_17, hidden_states_40], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_97, reinterpret_tensor(buf111, (8, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf113)
        del primals_97
        buf114 = buf103; del buf103  # reuse
        buf115 = buf102; del buf102  # reuse
        buf117 = reinterpret_tensor(buf113, (1, 8, 768), (6144, 768, 1), 0); del buf113  # reuse
        buf118 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf262 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_16(buf117, buf106, primals_98, primals_99, buf114, buf115, buf118, buf262)
        del primals_99
        buf119 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_42, hidden_states_43], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_101, reinterpret_tensor(buf118, (8, 768), (768, 1), 0), reinterpret_tensor(primals_100, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf119)
        del primals_101
        buf120 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_17(buf119, buf120)
        buf121 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_43, hidden_states_44, hidden_states_45], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_103, reinterpret_tensor(buf120, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_102, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf121)
        del primals_103
        buf122 = buf115; del buf115  # reuse
        buf123 = buf114; del buf114  # reuse
        buf125 = reinterpret_tensor(buf121, (1, 8, 768), (6144, 768, 1), 0); del buf121  # reuse
        buf126 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf261 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_18(buf125, buf118, primals_104, primals_105, buf122, buf123, buf126, buf261)
        del primals_105
        buf127 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_47, linear_36], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_107, reinterpret_tensor(buf126, (8, 768), (768, 1), 0), reinterpret_tensor(primals_106, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf127)
        del primals_107
        buf128 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_47, linear_36, linear_37], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_109, reinterpret_tensor(buf126, (8, 768), (768, 1), 0), reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf128)
        del primals_109
        buf129 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_47, linear_36, linear_38], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_111, reinterpret_tensor(buf126, (8, 768), (768, 1), 0), reinterpret_tensor(primals_110, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf129)
        del primals_111
        # Topologically Sorted Source Nodes: [linear_36, view_18, query_layer_6, linear_37, view_19, key_layer_6, linear_38, view_20, value_layer_6, attn_output_18], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf130 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf127, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf128, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf129, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf131 = buf130[0]
        assert_size_stride(buf131, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf131, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf132 = buf130[1]
        assert_size_stride(buf132, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf132, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf130
        buf133 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_19, attn_output_20, hidden_states_48], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_113, reinterpret_tensor(buf131, (8, 768), (768, 1), 0), reinterpret_tensor(primals_112, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del primals_113
        buf134 = buf123; del buf123  # reuse
        buf135 = buf122; del buf122  # reuse
        buf137 = reinterpret_tensor(buf133, (1, 8, 768), (6144, 768, 1), 0); del buf133  # reuse
        buf138 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf260 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_19(buf137, buf126, primals_114, primals_115, buf134, buf135, buf138, buf260)
        del primals_115
        buf139 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_50, hidden_states_51], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_117, reinterpret_tensor(buf138, (8, 768), (768, 1), 0), reinterpret_tensor(primals_116, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf139)
        del primals_117
        buf140 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_20(buf139, buf140)
        buf141 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_51, hidden_states_52, hidden_states_53], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_119, reinterpret_tensor(buf140, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_118, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf141)
        del primals_119
        buf142 = buf135; del buf135  # reuse
        buf143 = buf134; del buf134  # reuse
        buf145 = reinterpret_tensor(buf141, (1, 8, 768), (6144, 768, 1), 0); del buf141  # reuse
        buf146 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf259 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_21(buf145, buf138, primals_120, primals_121, buf142, buf143, buf146, buf259)
        del primals_121
        buf147 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_55, linear_42], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_123, reinterpret_tensor(buf146, (8, 768), (768, 1), 0), reinterpret_tensor(primals_122, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf147)
        del primals_123
        buf148 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_55, linear_42, linear_43], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_125, reinterpret_tensor(buf146, (8, 768), (768, 1), 0), reinterpret_tensor(primals_124, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf148)
        del primals_125
        buf149 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_55, linear_42, linear_44], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_127, reinterpret_tensor(buf146, (8, 768), (768, 1), 0), reinterpret_tensor(primals_126, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf149)
        del primals_127
        # Topologically Sorted Source Nodes: [linear_42, view_21, query_layer_7, linear_43, view_22, key_layer_7, linear_44, view_23, value_layer_7, attn_output_21], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf150 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf147, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf148, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf149, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf151 = buf150[0]
        assert_size_stride(buf151, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf151, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf152 = buf150[1]
        assert_size_stride(buf152, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf152, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf150
        buf153 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_22, attn_output_23, hidden_states_56], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_129, reinterpret_tensor(buf151, (8, 768), (768, 1), 0), reinterpret_tensor(primals_128, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf153)
        del primals_129
        buf154 = buf143; del buf143  # reuse
        buf155 = buf142; del buf142  # reuse
        buf157 = reinterpret_tensor(buf153, (1, 8, 768), (6144, 768, 1), 0); del buf153  # reuse
        buf158 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf258 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_22(buf157, buf146, primals_130, primals_131, buf154, buf155, buf158, buf258)
        del primals_131
        buf159 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_133, reinterpret_tensor(buf158, (8, 768), (768, 1), 0), reinterpret_tensor(primals_132, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf159)
        del primals_133
        buf160 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_23(buf159, buf160)
        buf161 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_59, hidden_states_60, hidden_states_61], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_135, reinterpret_tensor(buf160, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_134, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf161)
        del primals_135
        buf162 = buf155; del buf155  # reuse
        buf163 = buf154; del buf154  # reuse
        buf165 = reinterpret_tensor(buf161, (1, 8, 768), (6144, 768, 1), 0); del buf161  # reuse
        buf166 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf257 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_24(buf165, buf158, primals_136, primals_137, buf162, buf163, buf166, buf257)
        del primals_137
        buf167 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_63, linear_48], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_139, reinterpret_tensor(buf166, (8, 768), (768, 1), 0), reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
        del primals_139
        buf168 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_63, linear_48, linear_49], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_141, reinterpret_tensor(buf166, (8, 768), (768, 1), 0), reinterpret_tensor(primals_140, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
        del primals_141
        buf169 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_63, linear_48, linear_50], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_143, reinterpret_tensor(buf166, (8, 768), (768, 1), 0), reinterpret_tensor(primals_142, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf169)
        del primals_143
        # Topologically Sorted Source Nodes: [linear_48, view_24, query_layer_8, linear_49, view_25, key_layer_8, linear_50, view_26, value_layer_8, attn_output_24], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf170 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf167, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf168, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf169, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf171 = buf170[0]
        assert_size_stride(buf171, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf171, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf172 = buf170[1]
        assert_size_stride(buf172, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf172, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf170
        buf173 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_25, attn_output_26, hidden_states_64], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_145, reinterpret_tensor(buf171, (8, 768), (768, 1), 0), reinterpret_tensor(primals_144, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
        del primals_145
        buf174 = buf163; del buf163  # reuse
        buf175 = buf162; del buf162  # reuse
        buf177 = reinterpret_tensor(buf173, (1, 8, 768), (6144, 768, 1), 0); del buf173  # reuse
        buf178 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf256 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_25(buf177, buf166, primals_146, primals_147, buf174, buf175, buf178, buf256)
        del primals_147
        buf179 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_66, hidden_states_67], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_149, reinterpret_tensor(buf178, (8, 768), (768, 1), 0), reinterpret_tensor(primals_148, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf179)
        del primals_149
        buf180 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_26(buf179, buf180)
        buf181 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_67, hidden_states_68, hidden_states_69], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_151, reinterpret_tensor(buf180, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_150, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf181)
        del primals_151
        buf182 = buf175; del buf175  # reuse
        buf183 = buf174; del buf174  # reuse
        buf185 = reinterpret_tensor(buf181, (1, 8, 768), (6144, 768, 1), 0); del buf181  # reuse
        buf186 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf255 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_27(buf185, buf178, primals_152, primals_153, buf182, buf183, buf186, buf255)
        del primals_153
        buf187 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_71, linear_54], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_155, reinterpret_tensor(buf186, (8, 768), (768, 1), 0), reinterpret_tensor(primals_154, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf187)
        del primals_155
        buf188 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_71, linear_54, linear_55], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_157, reinterpret_tensor(buf186, (8, 768), (768, 1), 0), reinterpret_tensor(primals_156, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf188)
        del primals_157
        buf189 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_71, linear_54, linear_56], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_159, reinterpret_tensor(buf186, (8, 768), (768, 1), 0), reinterpret_tensor(primals_158, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf189)
        del primals_159
        # Topologically Sorted Source Nodes: [linear_54, view_27, query_layer_9, linear_55, view_28, key_layer_9, linear_56, view_29, value_layer_9, attn_output_27], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf190 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf187, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf188, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf189, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf191 = buf190[0]
        assert_size_stride(buf191, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf191, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf192 = buf190[1]
        assert_size_stride(buf192, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf192, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf190
        buf193 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_28, attn_output_29, hidden_states_72], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_161, reinterpret_tensor(buf191, (8, 768), (768, 1), 0), reinterpret_tensor(primals_160, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf193)
        del primals_161
        buf194 = buf183; del buf183  # reuse
        buf195 = buf182; del buf182  # reuse
        buf197 = reinterpret_tensor(buf193, (1, 8, 768), (6144, 768, 1), 0); del buf193  # reuse
        buf198 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf254 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_28(buf197, buf186, primals_162, primals_163, buf194, buf195, buf198, buf254)
        del primals_163
        buf199 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_74, hidden_states_75], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_165, reinterpret_tensor(buf198, (8, 768), (768, 1), 0), reinterpret_tensor(primals_164, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del primals_165
        buf200 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_29(buf199, buf200)
        buf201 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_75, hidden_states_76, hidden_states_77], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_167, reinterpret_tensor(buf200, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_166, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf201)
        del primals_167
        buf202 = buf195; del buf195  # reuse
        buf203 = buf194; del buf194  # reuse
        buf205 = reinterpret_tensor(buf201, (1, 8, 768), (6144, 768, 1), 0); del buf201  # reuse
        buf206 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf253 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_30(buf205, buf198, primals_168, primals_169, buf202, buf203, buf206, buf253)
        del primals_169
        buf207 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_79, linear_60], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_171, reinterpret_tensor(buf206, (8, 768), (768, 1), 0), reinterpret_tensor(primals_170, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf207)
        del primals_171
        buf208 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_79, linear_60, linear_61], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_173, reinterpret_tensor(buf206, (8, 768), (768, 1), 0), reinterpret_tensor(primals_172, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf208)
        del primals_173
        buf209 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_79, linear_60, linear_62], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_175, reinterpret_tensor(buf206, (8, 768), (768, 1), 0), reinterpret_tensor(primals_174, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf209)
        del primals_175
        # Topologically Sorted Source Nodes: [linear_60, view_30, query_layer_10, linear_61, view_31, key_layer_10, linear_62, view_32, value_layer_10, attn_output_30], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf210 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf207, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf208, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf209, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf211 = buf210[0]
        assert_size_stride(buf211, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf211, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf212 = buf210[1]
        assert_size_stride(buf212, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf212, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf210
        buf213 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_31, attn_output_32, hidden_states_80], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_177, reinterpret_tensor(buf211, (8, 768), (768, 1), 0), reinterpret_tensor(primals_176, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf213)
        del primals_177
        buf214 = buf203; del buf203  # reuse
        buf215 = buf202; del buf202  # reuse
        buf217 = reinterpret_tensor(buf213, (1, 8, 768), (6144, 768, 1), 0); del buf213  # reuse
        buf218 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf252 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_31(buf217, buf206, primals_178, primals_179, buf214, buf215, buf218, buf252)
        del primals_179
        buf219 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_82, hidden_states_83], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_181, reinterpret_tensor(buf218, (8, 768), (768, 1), 0), reinterpret_tensor(primals_180, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf219)
        del primals_181
        buf220 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_32(buf219, buf220)
        buf221 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_83, hidden_states_84, hidden_states_85], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_183, reinterpret_tensor(buf220, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_182, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf221)
        del primals_183
        buf222 = buf215; del buf215  # reuse
        buf223 = buf214; del buf214  # reuse
        buf225 = reinterpret_tensor(buf221, (1, 8, 768), (6144, 768, 1), 0); del buf221  # reuse
        buf226 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf251 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_33(buf225, buf218, primals_184, primals_185, buf222, buf223, buf226, buf251)
        del primals_185
        buf227 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_87, linear_66], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_187, reinterpret_tensor(buf226, (8, 768), (768, 1), 0), reinterpret_tensor(primals_186, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
        del primals_187
        buf228 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_87, linear_66, linear_67], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_189, reinterpret_tensor(buf226, (8, 768), (768, 1), 0), reinterpret_tensor(primals_188, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf228)
        del primals_189
        buf229 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_87, linear_66, linear_68], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_191, reinterpret_tensor(buf226, (8, 768), (768, 1), 0), reinterpret_tensor(primals_190, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf229)
        del primals_191
        # Topologically Sorted Source Nodes: [linear_66, view_33, query_layer_11, linear_67, view_34, key_layer_11, linear_68, view_35, value_layer_11, attn_output_33], Original ATen: [aten.view, aten.transpose, aten._scaled_dot_product_flash_attention_for_cpu]
        buf230 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(reinterpret_tensor(buf227, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf228, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf229, (1, 12, 8, 64), (6144, 64, 768, 1), 0), attn_mask=buf5)
        buf231 = buf230[0]
        assert_size_stride(buf231, (1, 12, 8, 64), (6144, 64, 768, 1), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf231, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        buf232 = buf230[1]
        assert_size_stride(buf232, (1, 12, 8), (96, 1, 12), 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        assert_alignment(buf232, 16, 'torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default')
        del buf230
        buf233 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_output_34, attn_output_35, hidden_states_88], Original ATen: [aten.transpose, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_193, reinterpret_tensor(buf231, (8, 768), (768, 1), 0), reinterpret_tensor(primals_192, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf233)
        del primals_193
        buf234 = buf223; del buf223  # reuse
        buf235 = buf222; del buf222  # reuse
        buf237 = reinterpret_tensor(buf233, (1, 8, 768), (6144, 768, 1), 0); del buf233  # reuse
        buf238 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf250 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_34(buf237, buf226, primals_194, primals_195, buf234, buf235, buf238, buf250)
        del primals_195
        buf239 = empty_strided_cpu((8, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_90, hidden_states_91], Original ATen: [aten.native_layer_norm, aten.view, aten.t, aten.addmm]
        extern_kernels.addmm(primals_197, reinterpret_tensor(buf238, (8, 768), (768, 1), 0), reinterpret_tensor(primals_196, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf239)
        del primals_197
        buf240 = empty_strided_cpu((1, 8, 3072), (24576, 3072, 1), torch.float32)
        cpp_fused_gelu_view_35(buf239, buf240)
        buf241 = empty_strided_cpu((8, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92, hidden_states_93], Original ATen: [aten.view, aten.gelu, aten.t, aten.addmm]
        extern_kernels.addmm(primals_199, reinterpret_tensor(buf240, (8, 3072), (3072, 1), 0), reinterpret_tensor(primals_198, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf241)
        del primals_199
        buf242 = buf235; del buf235  # reuse
        buf243 = buf234; del buf234  # reuse
        buf245 = reinterpret_tensor(buf241, (1, 8, 768), (6144, 768, 1), 0); del buf241  # reuse
        buf246 = empty_strided_cpu((1, 8, 768), (6144, 768, 1), torch.float32)
        buf249 = empty_strided_cpu((1, 8, 1), (8, 1, 1), torch.float32)
        cpp_fused_add_native_layer_norm_native_layer_norm_backward_view_36(buf245, buf238, primals_200, primals_201, buf242, buf243, buf246, buf249)
        del buf242
        del buf243
        del primals_201
        buf247 = empty_strided_cpu((1, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [first_token_tensor, pooled_output], Original ATen: [aten.select, aten.t, aten.addmm]
        extern_kernels.addmm(primals_203, reinterpret_tensor(buf246, (1, 768), (768, 1), 0), reinterpret_tensor(primals_202, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf247)
        del primals_203
        buf248 = buf247; del buf247  # reuse
        cpp_fused_tanh_37(buf248)
        return (buf246, buf248, primals_1, primals_2, primals_3, primals_7, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_118, primals_120, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, primals_156, primals_158, primals_160, primals_162, primals_164, primals_166, primals_168, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, buf4, buf5, reinterpret_tensor(buf6, (8, 768), (768, 1), 0), reinterpret_tensor(buf7, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf8, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf9, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf11, buf12, buf17, reinterpret_tensor(buf18, (8, 768), (768, 1), 0), buf19, reinterpret_tensor(buf20, (8, 3072), (3072, 1), 0), buf25, reinterpret_tensor(buf26, (8, 768), (768, 1), 0), reinterpret_tensor(buf27, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf28, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf29, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf31, buf32, buf37, reinterpret_tensor(buf38, (8, 768), (768, 1), 0), buf39, reinterpret_tensor(buf40, (8, 3072), (3072, 1), 0), buf45, reinterpret_tensor(buf46, (8, 768), (768, 1), 0), reinterpret_tensor(buf47, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf48, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf49, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf51, buf52, buf57, reinterpret_tensor(buf58, (8, 768), (768, 1), 0), buf59, reinterpret_tensor(buf60, (8, 3072), (3072, 1), 0), buf65, reinterpret_tensor(buf66, (8, 768), (768, 1), 0), reinterpret_tensor(buf67, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf68, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf69, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf71, buf72, buf77, reinterpret_tensor(buf78, (8, 768), (768, 1), 0), buf79, reinterpret_tensor(buf80, (8, 3072), (3072, 1), 0), buf85, reinterpret_tensor(buf86, (8, 768), (768, 1), 0), reinterpret_tensor(buf87, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf88, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf89, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf91, buf92, buf97, reinterpret_tensor(buf98, (8, 768), (768, 1), 0), buf99, reinterpret_tensor(buf100, (8, 3072), (3072, 1), 0), buf105, reinterpret_tensor(buf106, (8, 768), (768, 1), 0), reinterpret_tensor(buf107, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf108, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf109, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf111, buf112, buf117, reinterpret_tensor(buf118, (8, 768), (768, 1), 0), buf119, reinterpret_tensor(buf120, (8, 3072), (3072, 1), 0), buf125, reinterpret_tensor(buf126, (8, 768), (768, 1), 0), reinterpret_tensor(buf127, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf128, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf129, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf131, buf132, buf137, reinterpret_tensor(buf138, (8, 768), (768, 1), 0), buf139, reinterpret_tensor(buf140, (8, 3072), (3072, 1), 0), buf145, reinterpret_tensor(buf146, (8, 768), (768, 1), 0), reinterpret_tensor(buf147, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf148, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf149, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf151, buf152, buf157, reinterpret_tensor(buf158, (8, 768), (768, 1), 0), buf159, reinterpret_tensor(buf160, (8, 3072), (3072, 1), 0), buf165, reinterpret_tensor(buf166, (8, 768), (768, 1), 0), reinterpret_tensor(buf167, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf168, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf169, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf171, buf172, buf177, reinterpret_tensor(buf178, (8, 768), (768, 1), 0), buf179, reinterpret_tensor(buf180, (8, 3072), (3072, 1), 0), buf185, reinterpret_tensor(buf186, (8, 768), (768, 1), 0), reinterpret_tensor(buf187, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf188, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf189, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf191, buf192, buf197, reinterpret_tensor(buf198, (8, 768), (768, 1), 0), buf199, reinterpret_tensor(buf200, (8, 3072), (3072, 1), 0), buf205, reinterpret_tensor(buf206, (8, 768), (768, 1), 0), reinterpret_tensor(buf207, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf208, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf209, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf211, buf212, buf217, reinterpret_tensor(buf218, (8, 768), (768, 1), 0), buf219, reinterpret_tensor(buf220, (8, 3072), (3072, 1), 0), buf225, reinterpret_tensor(buf226, (8, 768), (768, 1), 0), reinterpret_tensor(buf227, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf228, (1, 12, 8, 64), (6144, 64, 768, 1), 0), reinterpret_tensor(buf229, (1, 12, 8, 64), (6144, 64, 768, 1), 0), buf231, buf232, buf237, reinterpret_tensor(buf238, (8, 768), (768, 1), 0), buf239, reinterpret_tensor(buf240, (8, 3072), (3072, 1), 0), buf245, reinterpret_tensor(buf246, (1, 768), (6144, 1), 0), buf248, buf249, buf250, buf251, buf252, buf253, buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf273, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 8), (8, 1), device='cpu', dtype=torch.int64)
    primals_2 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_3 = rand_strided((1, 512), (512, 1), device='cpu', dtype=torch.int64)
    primals_4 = rand_strided((30522, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((2, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((512, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((1, 8), (8, 1), device='cpu', dtype=torch.int64)
    primals_10 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
