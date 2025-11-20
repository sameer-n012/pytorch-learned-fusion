
data/torch_compile_debug/bert-base-uncased/triton_kernels/YU2JAVMUDZ5H5NM2UV7YLARYIS35PZL4WNAPRO3QLGCUQRK25FTA/cuda_utils.cpython-311-x86_64-linux-gnu.so:     file format elf64-x86-64


Disassembly of section .init:

0000000000002000 <_init>:
    2000:	f3 0f 1e fa          	endbr64 
    2004:	48 83 ec 08          	sub    $0x8,%rsp
    2008:	48 8b 05 d1 3f 00 00 	mov    0x3fd1(%rip),%rax        # 5fe0 <__gmon_start__@Base>
    200f:	48 85 c0             	test   %rax,%rax
    2012:	74 02                	je     2016 <_init+0x16>
    2014:	ff d0                	call   *%rax
    2016:	48 83 c4 08          	add    $0x8,%rsp
    201a:	c3                   	ret    

Disassembly of section .plt:

0000000000002020 <.plt>:
    2020:	ff 35 e2 3f 00 00    	push   0x3fe2(%rip)        # 6008 <_GLOBAL_OFFSET_TABLE_+0x8>
    2026:	f2 ff 25 e3 3f 00 00 	bnd jmp *0x3fe3(%rip)        # 6010 <_GLOBAL_OFFSET_TABLE_+0x10>
    202d:	0f 1f 00             	nopl   (%rax)
    2030:	f3 0f 1e fa          	endbr64 
    2034:	68 00 00 00 00       	push   $0x0
    2039:	f2 e9 e1 ff ff ff    	bnd jmp 2020 <_init+0x20>
    203f:	90                   	nop
    2040:	f3 0f 1e fa          	endbr64 
    2044:	68 01 00 00 00       	push   $0x1
    2049:	f2 e9 d1 ff ff ff    	bnd jmp 2020 <_init+0x20>
    204f:	90                   	nop
    2050:	f3 0f 1e fa          	endbr64 
    2054:	68 02 00 00 00       	push   $0x2
    2059:	f2 e9 c1 ff ff ff    	bnd jmp 2020 <_init+0x20>
    205f:	90                   	nop
    2060:	f3 0f 1e fa          	endbr64 
    2064:	68 03 00 00 00       	push   $0x3
    2069:	f2 e9 b1 ff ff ff    	bnd jmp 2020 <_init+0x20>
    206f:	90                   	nop
    2070:	f3 0f 1e fa          	endbr64 
    2074:	68 04 00 00 00       	push   $0x4
    2079:	f2 e9 a1 ff ff ff    	bnd jmp 2020 <_init+0x20>
    207f:	90                   	nop
    2080:	f3 0f 1e fa          	endbr64 
    2084:	68 05 00 00 00       	push   $0x5
    2089:	f2 e9 91 ff ff ff    	bnd jmp 2020 <_init+0x20>
    208f:	90                   	nop
    2090:	f3 0f 1e fa          	endbr64 
    2094:	68 06 00 00 00       	push   $0x6
    2099:	f2 e9 81 ff ff ff    	bnd jmp 2020 <_init+0x20>
    209f:	90                   	nop
    20a0:	f3 0f 1e fa          	endbr64 
    20a4:	68 07 00 00 00       	push   $0x7
    20a9:	f2 e9 71 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20af:	90                   	nop
    20b0:	f3 0f 1e fa          	endbr64 
    20b4:	68 08 00 00 00       	push   $0x8
    20b9:	f2 e9 61 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20bf:	90                   	nop
    20c0:	f3 0f 1e fa          	endbr64 
    20c4:	68 09 00 00 00       	push   $0x9
    20c9:	f2 e9 51 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20cf:	90                   	nop
    20d0:	f3 0f 1e fa          	endbr64 
    20d4:	68 0a 00 00 00       	push   $0xa
    20d9:	f2 e9 41 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20df:	90                   	nop
    20e0:	f3 0f 1e fa          	endbr64 
    20e4:	68 0b 00 00 00       	push   $0xb
    20e9:	f2 e9 31 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20ef:	90                   	nop
    20f0:	f3 0f 1e fa          	endbr64 
    20f4:	68 0c 00 00 00       	push   $0xc
    20f9:	f2 e9 21 ff ff ff    	bnd jmp 2020 <_init+0x20>
    20ff:	90                   	nop
    2100:	f3 0f 1e fa          	endbr64 
    2104:	68 0d 00 00 00       	push   $0xd
    2109:	f2 e9 11 ff ff ff    	bnd jmp 2020 <_init+0x20>
    210f:	90                   	nop
    2110:	f3 0f 1e fa          	endbr64 
    2114:	68 0e 00 00 00       	push   $0xe
    2119:	f2 e9 01 ff ff ff    	bnd jmp 2020 <_init+0x20>
    211f:	90                   	nop
    2120:	f3 0f 1e fa          	endbr64 
    2124:	68 0f 00 00 00       	push   $0xf
    2129:	f2 e9 f1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    212f:	90                   	nop
    2130:	f3 0f 1e fa          	endbr64 
    2134:	68 10 00 00 00       	push   $0x10
    2139:	f2 e9 e1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    213f:	90                   	nop
    2140:	f3 0f 1e fa          	endbr64 
    2144:	68 11 00 00 00       	push   $0x11
    2149:	f2 e9 d1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    214f:	90                   	nop
    2150:	f3 0f 1e fa          	endbr64 
    2154:	68 12 00 00 00       	push   $0x12
    2159:	f2 e9 c1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    215f:	90                   	nop
    2160:	f3 0f 1e fa          	endbr64 
    2164:	68 13 00 00 00       	push   $0x13
    2169:	f2 e9 b1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    216f:	90                   	nop
    2170:	f3 0f 1e fa          	endbr64 
    2174:	68 14 00 00 00       	push   $0x14
    2179:	f2 e9 a1 fe ff ff    	bnd jmp 2020 <_init+0x20>
    217f:	90                   	nop
    2180:	f3 0f 1e fa          	endbr64 
    2184:	68 15 00 00 00       	push   $0x15
    2189:	f2 e9 91 fe ff ff    	bnd jmp 2020 <_init+0x20>
    218f:	90                   	nop
    2190:	f3 0f 1e fa          	endbr64 
    2194:	68 16 00 00 00       	push   $0x16
    2199:	f2 e9 81 fe ff ff    	bnd jmp 2020 <_init+0x20>
    219f:	90                   	nop
    21a0:	f3 0f 1e fa          	endbr64 
    21a4:	68 17 00 00 00       	push   $0x17
    21a9:	f2 e9 71 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21af:	90                   	nop
    21b0:	f3 0f 1e fa          	endbr64 
    21b4:	68 18 00 00 00       	push   $0x18
    21b9:	f2 e9 61 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21bf:	90                   	nop
    21c0:	f3 0f 1e fa          	endbr64 
    21c4:	68 19 00 00 00       	push   $0x19
    21c9:	f2 e9 51 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21cf:	90                   	nop
    21d0:	f3 0f 1e fa          	endbr64 
    21d4:	68 1a 00 00 00       	push   $0x1a
    21d9:	f2 e9 41 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21df:	90                   	nop
    21e0:	f3 0f 1e fa          	endbr64 
    21e4:	68 1b 00 00 00       	push   $0x1b
    21e9:	f2 e9 31 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21ef:	90                   	nop
    21f0:	f3 0f 1e fa          	endbr64 
    21f4:	68 1c 00 00 00       	push   $0x1c
    21f9:	f2 e9 21 fe ff ff    	bnd jmp 2020 <_init+0x20>
    21ff:	90                   	nop
    2200:	f3 0f 1e fa          	endbr64 
    2204:	68 1d 00 00 00       	push   $0x1d
    2209:	f2 e9 11 fe ff ff    	bnd jmp 2020 <_init+0x20>
    220f:	90                   	nop
    2210:	f3 0f 1e fa          	endbr64 
    2214:	68 1e 00 00 00       	push   $0x1e
    2219:	f2 e9 01 fe ff ff    	bnd jmp 2020 <_init+0x20>
    221f:	90                   	nop
    2220:	f3 0f 1e fa          	endbr64 
    2224:	68 1f 00 00 00       	push   $0x1f
    2229:	f2 e9 f1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    222f:	90                   	nop
    2230:	f3 0f 1e fa          	endbr64 
    2234:	68 20 00 00 00       	push   $0x20
    2239:	f2 e9 e1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    223f:	90                   	nop
    2240:	f3 0f 1e fa          	endbr64 
    2244:	68 21 00 00 00       	push   $0x21
    2249:	f2 e9 d1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    224f:	90                   	nop
    2250:	f3 0f 1e fa          	endbr64 
    2254:	68 22 00 00 00       	push   $0x22
    2259:	f2 e9 c1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    225f:	90                   	nop
    2260:	f3 0f 1e fa          	endbr64 
    2264:	68 23 00 00 00       	push   $0x23
    2269:	f2 e9 b1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    226f:	90                   	nop
    2270:	f3 0f 1e fa          	endbr64 
    2274:	68 24 00 00 00       	push   $0x24
    2279:	f2 e9 a1 fd ff ff    	bnd jmp 2020 <_init+0x20>
    227f:	90                   	nop
    2280:	f3 0f 1e fa          	endbr64 
    2284:	68 25 00 00 00       	push   $0x25
    2289:	f2 e9 91 fd ff ff    	bnd jmp 2020 <_init+0x20>
    228f:	90                   	nop
    2290:	f3 0f 1e fa          	endbr64 
    2294:	68 26 00 00 00       	push   $0x26
    2299:	f2 e9 81 fd ff ff    	bnd jmp 2020 <_init+0x20>
    229f:	90                   	nop
    22a0:	f3 0f 1e fa          	endbr64 
    22a4:	68 27 00 00 00       	push   $0x27
    22a9:	f2 e9 71 fd ff ff    	bnd jmp 2020 <_init+0x20>
    22af:	90                   	nop
    22b0:	f3 0f 1e fa          	endbr64 
    22b4:	68 28 00 00 00       	push   $0x28
    22b9:	f2 e9 61 fd ff ff    	bnd jmp 2020 <_init+0x20>
    22bf:	90                   	nop
    22c0:	f3 0f 1e fa          	endbr64 
    22c4:	68 29 00 00 00       	push   $0x29
    22c9:	f2 e9 51 fd ff ff    	bnd jmp 2020 <_init+0x20>
    22cf:	90                   	nop

Disassembly of section .plt.got:

00000000000022d0 <__cxa_finalize@plt>:
    22d0:	f3 0f 1e fa          	endbr64 
    22d4:	f2 ff 25 1d 3d 00 00 	bnd jmp *0x3d1d(%rip)        # 5ff8 <__cxa_finalize@GLIBC_2.2.5>
    22db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000022e0 <cuModuleGetFunction@plt>:
    22e0:	f3 0f 1e fa          	endbr64 
    22e4:	f2 ff 25 2d 3d 00 00 	bnd jmp *0x3d2d(%rip)        # 6018 <cuModuleGetFunction@Base>
    22eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000022f0 <cuFuncSetAttribute@plt>:
    22f0:	f3 0f 1e fa          	endbr64 
    22f4:	f2 ff 25 25 3d 00 00 	bnd jmp *0x3d25(%rip)        # 6020 <cuFuncSetAttribute@Base>
    22fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002300 <PyObject_Init@plt>:
    2300:	f3 0f 1e fa          	endbr64 
    2304:	f2 ff 25 1d 3d 00 00 	bnd jmp *0x3d1d(%rip)        # 6028 <PyObject_Init@Base>
    230b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002310 <cuModuleLoadData@plt>:
    2310:	f3 0f 1e fa          	endbr64 
    2314:	f2 ff 25 15 3d 00 00 	bnd jmp *0x3d15(%rip)        # 6030 <cuModuleLoadData@Base>
    231b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002320 <dlerror@plt>:
    2320:	f3 0f 1e fa          	endbr64 
    2324:	f2 ff 25 0d 3d 00 00 	bnd jmp *0x3d0d(%rip)        # 6038 <dlerror@GLIBC_2.34>
    232b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002330 <free@plt>:
    2330:	f3 0f 1e fa          	endbr64 
    2334:	f2 ff 25 05 3d 00 00 	bnd jmp *0x3d05(%rip)        # 6040 <free@GLIBC_2.2.5>
    233b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002340 <cuFuncGetAttribute@plt>:
    2340:	f3 0f 1e fa          	endbr64 
    2344:	f2 ff 25 fd 3c 00 00 	bnd jmp *0x3cfd(%rip)        # 6048 <cuFuncGetAttribute@Base>
    234b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002350 <cuCtxSetCurrent@plt>:
    2350:	f3 0f 1e fa          	endbr64 
    2354:	f2 ff 25 f5 3c 00 00 	bnd jmp *0x3cf5(%rip)        # 6050 <cuCtxSetCurrent@Base>
    235b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002360 <PyGILState_Release@plt>:
    2360:	f3 0f 1e fa          	endbr64 
    2364:	f2 ff 25 ed 3c 00 00 	bnd jmp *0x3ced(%rip)        # 6058 <PyGILState_Release@Base>
    236b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002370 <PyEval_RestoreThread@plt>:
    2370:	f3 0f 1e fa          	endbr64 
    2374:	f2 ff 25 e5 3c 00 00 	bnd jmp *0x3ce5(%rip)        # 6060 <PyEval_RestoreThread@Base>
    237b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002380 <_Py_Dealloc@plt>:
    2380:	f3 0f 1e fa          	endbr64 
    2384:	f2 ff 25 dd 3c 00 00 	bnd jmp *0x3cdd(%rip)        # 6068 <_Py_Dealloc@Base>
    238b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002390 <PyModule_AddFunctions@plt>:
    2390:	f3 0f 1e fa          	endbr64 
    2394:	f2 ff 25 d5 3c 00 00 	bnd jmp *0x3cd5(%rip)        # 6070 <PyModule_AddFunctions@Base>
    239b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023a0 <PyErr_NoMemory@plt>:
    23a0:	f3 0f 1e fa          	endbr64 
    23a4:	f2 ff 25 cd 3c 00 00 	bnd jmp *0x3ccd(%rip)        # 6078 <PyErr_NoMemory@Base>
    23ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023b0 <__stack_chk_fail@plt>:
    23b0:	f3 0f 1e fa          	endbr64 
    23b4:	f2 ff 25 c5 3c 00 00 	bnd jmp *0x3cc5(%rip)        # 6080 <__stack_chk_fail@GLIBC_2.4>
    23bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023c0 <PyErr_SetString@plt>:
    23c0:	f3 0f 1e fa          	endbr64 
    23c4:	f2 ff 25 bd 3c 00 00 	bnd jmp *0x3cbd(%rip)        # 6088 <PyErr_SetString@Base>
    23cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023d0 <__assert_fail@plt>:
    23d0:	f3 0f 1e fa          	endbr64 
    23d4:	f2 ff 25 b5 3c 00 00 	bnd jmp *0x3cb5(%rip)        # 6090 <__assert_fail@GLIBC_2.2.5>
    23db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023e0 <PyGILState_Ensure@plt>:
    23e0:	f3 0f 1e fa          	endbr64 
    23e4:	f2 ff 25 ad 3c 00 00 	bnd jmp *0x3cad(%rip)        # 6098 <PyGILState_Ensure@Base>
    23eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000023f0 <cuDeviceGet@plt>:
    23f0:	f3 0f 1e fa          	endbr64 
    23f4:	f2 ff 25 a5 3c 00 00 	bnd jmp *0x3ca5(%rip)        # 60a0 <cuDeviceGet@Base>
    23fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002400 <PyType_Ready@plt>:
    2400:	f3 0f 1e fa          	endbr64 
    2404:	f2 ff 25 9d 3c 00 00 	bnd jmp *0x3c9d(%rip)        # 60a8 <PyType_Ready@Base>
    240b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002410 <PyLong_FromLong@plt>:
    2410:	f3 0f 1e fa          	endbr64 
    2414:	f2 ff 25 95 3c 00 00 	bnd jmp *0x3c95(%rip)        # 60b0 <PyLong_FromLong@Base>
    241b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002420 <dlopen@plt>:
    2420:	f3 0f 1e fa          	endbr64 
    2424:	f2 ff 25 8d 3c 00 00 	bnd jmp *0x3c8d(%rip)        # 60b8 <dlopen@GLIBC_2.34>
    242b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002430 <cuDevicePrimaryCtxRetain@plt>:
    2430:	f3 0f 1e fa          	endbr64 
    2434:	f2 ff 25 85 3c 00 00 	bnd jmp *0x3c85(%rip)        # 60c0 <cuDevicePrimaryCtxRetain@Base>
    243b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002440 <PyErr_Occurred@plt>:
    2440:	f3 0f 1e fa          	endbr64 
    2444:	f2 ff 25 7d 3c 00 00 	bnd jmp *0x3c7d(%rip)        # 60c8 <PyErr_Occurred@Base>
    244b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002450 <PyModule_Create2@plt>:
    2450:	f3 0f 1e fa          	endbr64 
    2454:	f2 ff 25 75 3c 00 00 	bnd jmp *0x3c75(%rip)        # 60d0 <PyModule_Create2@Base>
    245b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002460 <PyLong_AsLong@plt>:
    2460:	f3 0f 1e fa          	endbr64 
    2464:	f2 ff 25 6d 3c 00 00 	bnd jmp *0x3c6d(%rip)        # 60d8 <PyLong_AsLong@Base>
    246b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002470 <PyObject_CallObject@plt>:
    2470:	f3 0f 1e fa          	endbr64 
    2474:	f2 ff 25 65 3c 00 00 	bnd jmp *0x3c65(%rip)        # 60e0 <PyObject_CallObject@Base>
    247b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002480 <_PyArg_ParseTuple_SizeT@plt>:
    2480:	f3 0f 1e fa          	endbr64 
    2484:	f2 ff 25 5d 3c 00 00 	bnd jmp *0x3c5d(%rip)        # 60e8 <_PyArg_ParseTuple_SizeT@Base>
    248b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002490 <cuCtxSetLimit@plt>:
    2490:	f3 0f 1e fa          	endbr64 
    2494:	f2 ff 25 55 3c 00 00 	bnd jmp *0x3c55(%rip)        # 60f0 <cuCtxSetLimit@Base>
    249b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024a0 <cuGetErrorString@plt>:
    24a0:	f3 0f 1e fa          	endbr64 
    24a4:	f2 ff 25 4d 3c 00 00 	bnd jmp *0x3c4d(%rip)        # 60f8 <cuGetErrorString@Base>
    24ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024b0 <__strcpy_chk@plt>:
    24b0:	f3 0f 1e fa          	endbr64 
    24b4:	f2 ff 25 45 3c 00 00 	bnd jmp *0x3c45(%rip)        # 6100 <__strcpy_chk@GLIBC_2.3.4>
    24bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024c0 <PyEval_SaveThread@plt>:
    24c0:	f3 0f 1e fa          	endbr64 
    24c4:	f2 ff 25 3d 3c 00 00 	bnd jmp *0x3c3d(%rip)        # 6108 <PyEval_SaveThread@Base>
    24cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024d0 <PyModule_AddObject@plt>:
    24d0:	f3 0f 1e fa          	endbr64 
    24d4:	f2 ff 25 35 3c 00 00 	bnd jmp *0x3c35(%rip)        # 6110 <PyModule_AddObject@Base>
    24db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024e0 <dlsym@plt>:
    24e0:	f3 0f 1e fa          	endbr64 
    24e4:	f2 ff 25 2d 3c 00 00 	bnd jmp *0x3c2d(%rip)        # 6118 <dlsym@GLIBC_2.34>
    24eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000024f0 <PyLong_AsLongLong@plt>:
    24f0:	f3 0f 1e fa          	endbr64 
    24f4:	f2 ff 25 25 3c 00 00 	bnd jmp *0x3c25(%rip)        # 6120 <PyLong_AsLongLong@Base>
    24fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002500 <cuDeviceGetAttribute@plt>:
    2500:	f3 0f 1e fa          	endbr64 
    2504:	f2 ff 25 1d 3c 00 00 	bnd jmp *0x3c1d(%rip)        # 6128 <cuDeviceGetAttribute@Base>
    250b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002510 <cuCtxGetCurrent@plt>:
    2510:	f3 0f 1e fa          	endbr64 
    2514:	f2 ff 25 15 3c 00 00 	bnd jmp *0x3c15(%rip)        # 6130 <cuCtxGetCurrent@Base>
    251b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002520 <_Py_BuildValue_SizeT@plt>:
    2520:	f3 0f 1e fa          	endbr64 
    2524:	f2 ff 25 0d 3c 00 00 	bnd jmp *0x3c0d(%rip)        # 6138 <_Py_BuildValue_SizeT@Base>
    252b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002530 <posix_memalign@plt>:
    2530:	f3 0f 1e fa          	endbr64 
    2534:	f2 ff 25 05 3c 00 00 	bnd jmp *0x3c05(%rip)        # 6140 <posix_memalign@GLIBC_2.2.5>
    253b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002540 <cuFuncSetCacheConfig@plt>:
    2540:	f3 0f 1e fa          	endbr64 
    2544:	f2 ff 25 fd 3b 00 00 	bnd jmp *0x3bfd(%rip)        # 6148 <cuFuncSetCacheConfig@Base>
    254b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002550 <dlclose@plt>:
    2550:	f3 0f 1e fa          	endbr64 
    2554:	f2 ff 25 f5 3b 00 00 	bnd jmp *0x3bf5(%rip)        # 6150 <dlclose@GLIBC_2.34>
    255b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002560 <cuCtxGetLimit@plt>:
    2560:	f3 0f 1e fa          	endbr64 
    2564:	f2 ff 25 ed 3b 00 00 	bnd jmp *0x3bed(%rip)        # 6158 <cuCtxGetLimit@Base>
    256b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002570 <PySequence_Fast@plt>:
    2570:	f3 0f 1e fa          	endbr64 
    2574:	f2 ff 25 e5 3b 00 00 	bnd jmp *0x3be5(%rip)        # 6160 <PySequence_Fast@Base>
    257b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000002580 <deregister_tm_clones>:
    2580:	48 8d 3d f1 3e 00 00 	lea    0x3ef1(%rip),%rdi        # 6478 <completed.0>
    2587:	48 8d 05 ea 3e 00 00 	lea    0x3eea(%rip),%rax        # 6478 <completed.0>
    258e:	48 39 f8             	cmp    %rdi,%rax
    2591:	74 15                	je     25a8 <deregister_tm_clones+0x28>
    2593:	48 8b 05 26 3a 00 00 	mov    0x3a26(%rip),%rax        # 5fc0 <_ITM_deregisterTMCloneTable@Base>
    259a:	48 85 c0             	test   %rax,%rax
    259d:	74 09                	je     25a8 <deregister_tm_clones+0x28>
    259f:	ff e0                	jmp    *%rax
    25a1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    25a8:	c3                   	ret    
    25a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000025b0 <register_tm_clones>:
    25b0:	48 8d 3d c1 3e 00 00 	lea    0x3ec1(%rip),%rdi        # 6478 <completed.0>
    25b7:	48 8d 35 ba 3e 00 00 	lea    0x3eba(%rip),%rsi        # 6478 <completed.0>
    25be:	48 29 fe             	sub    %rdi,%rsi
    25c1:	48 89 f0             	mov    %rsi,%rax
    25c4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    25c8:	48 c1 f8 03          	sar    $0x3,%rax
    25cc:	48 01 c6             	add    %rax,%rsi
    25cf:	48 d1 fe             	sar    %rsi
    25d2:	74 14                	je     25e8 <register_tm_clones+0x38>
    25d4:	48 8b 05 15 3a 00 00 	mov    0x3a15(%rip),%rax        # 5ff0 <_ITM_registerTMCloneTable@Base>
    25db:	48 85 c0             	test   %rax,%rax
    25de:	74 08                	je     25e8 <register_tm_clones+0x38>
    25e0:	ff e0                	jmp    *%rax
    25e2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    25e8:	c3                   	ret    
    25e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000025f0 <__do_global_dtors_aux>:
    25f0:	f3 0f 1e fa          	endbr64 
    25f4:	80 3d 7d 3e 00 00 00 	cmpb   $0x0,0x3e7d(%rip)        # 6478 <completed.0>
    25fb:	75 2b                	jne    2628 <__do_global_dtors_aux+0x38>
    25fd:	55                   	push   %rbp
    25fe:	48 83 3d f2 39 00 00 	cmpq   $0x0,0x39f2(%rip)        # 5ff8 <__cxa_finalize@GLIBC_2.2.5>
    2605:	00 
    2606:	48 89 e5             	mov    %rsp,%rbp
    2609:	74 0c                	je     2617 <__do_global_dtors_aux+0x27>
    260b:	48 8b 3d 6e 3b 00 00 	mov    0x3b6e(%rip),%rdi        # 6180 <__dso_handle>
    2612:	e8 b9 fc ff ff       	call   22d0 <__cxa_finalize@plt>
    2617:	e8 64 ff ff ff       	call   2580 <deregister_tm_clones>
    261c:	c6 05 55 3e 00 00 01 	movb   $0x1,0x3e55(%rip)        # 6478 <completed.0>
    2623:	5d                   	pop    %rbp
    2624:	c3                   	ret    
    2625:	0f 1f 00             	nopl   (%rax)
    2628:	c3                   	ret    
    2629:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002630 <frame_dummy>:
    2630:	f3 0f 1e fa          	endbr64 
    2634:	e9 77 ff ff ff       	jmp    25b0 <register_tm_clones>
    2639:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002640 <PyCUtensorMap_dealloc>:
    2640:	f3 0f 1e fa          	endbr64 
    2644:	48 8b 47 08          	mov    0x8(%rdi),%rax
    2648:	ff a0 40 01 00 00    	jmp    *0x140(%rax)
    264e:	66 90                	xchg   %ax,%ax

0000000000002650 <PyCUtensorMap_free>:
    2650:	f3 0f 1e fa          	endbr64 
    2654:	e9 d7 fc ff ff       	jmp    2330 <free@plt>
    2659:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002660 <PyCUtensorMap_alloc>:
    2660:	f3 0f 1e fa          	endbr64 
    2664:	41 54                	push   %r12
    2666:	be 80 00 00 00       	mov    $0x80,%esi
    266b:	55                   	push   %rbp
    266c:	48 89 fd             	mov    %rdi,%rbp
    266f:	48 83 ec 18          	sub    $0x18,%rsp
    2673:	48 8b 57 20          	mov    0x20(%rdi),%rdx
    2677:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    267e:	00 00 
    2680:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    2685:	31 c0                	xor    %eax,%eax
    2687:	48 89 e7             	mov    %rsp,%rdi
    268a:	e8 a1 fe ff ff       	call   2530 <posix_memalign@plt>
    268f:	85 c0                	test   %eax,%eax
    2691:	74 2d                	je     26c0 <PyCUtensorMap_alloc+0x60>
    2693:	e8 08 fd ff ff       	call   23a0 <PyErr_NoMemory@plt>
    2698:	45 31 e4             	xor    %r12d,%r12d
    269b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    26a0:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    26a5:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    26ac:	00 00 
    26ae:	75 21                	jne    26d1 <PyCUtensorMap_alloc+0x71>
    26b0:	48 83 c4 18          	add    $0x18,%rsp
    26b4:	4c 89 e0             	mov    %r12,%rax
    26b7:	5d                   	pop    %rbp
    26b8:	41 5c                	pop    %r12
    26ba:	c3                   	ret    
    26bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    26c0:	4c 8b 24 24          	mov    (%rsp),%r12
    26c4:	48 89 ee             	mov    %rbp,%rsi
    26c7:	4c 89 e7             	mov    %r12,%rdi
    26ca:	e8 31 fc ff ff       	call   2300 <PyObject_Init@plt>
    26cf:	eb cf                	jmp    26a0 <PyCUtensorMap_alloc+0x40>
    26d1:	e8 da fc ff ff       	call   23b0 <__stack_chk_fail@plt>
    26d6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    26dd:	00 00 00 

00000000000026e0 <PyTuple_GET_SIZE.part.0>:
    26e0:	50                   	push   %rax
    26e1:	58                   	pop    %rax
    26e2:	48 8d 0d 77 1e 00 00 	lea    0x1e77(%rip),%rcx        # 4560 <__PRETTY_FUNCTION__.1>
    26e9:	ba 17 00 00 00       	mov    $0x17,%edx
    26ee:	48 8d 35 0b 19 00 00 	lea    0x190b(%rip),%rsi        # 4000 <_fini+0x620>
    26f5:	48 8d 3d f9 1b 00 00 	lea    0x1bf9(%rip),%rdi        # 42f5 <_fini+0x915>
    26fc:	48 83 ec 08          	sub    $0x8,%rsp
    2700:	e8 cb fc ff ff       	call   23d0 <__assert_fail@plt>
    2705:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    270c:	00 00 00 00 

0000000000002710 <gpuAssert.part.0>:
    2710:	41 54                	push   %r12
    2712:	55                   	push   %rbp
    2713:	48 81 ec 28 04 00 00 	sub    $0x428,%rsp
    271a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2721:	00 00 
    2723:	48 89 84 24 18 04 00 	mov    %rax,0x418(%rsp)
    272a:	00 
    272b:	31 c0                	xor    %eax,%eax
    272d:	48 8d 74 24 08       	lea    0x8(%rsp),%rsi
    2732:	4c 8d 64 24 10       	lea    0x10(%rsp),%r12
    2737:	e8 64 fd ff ff       	call   24a0 <cuGetErrorString@plt>
    273c:	66 0f 6f 05 5c 1e 00 	movdqa 0x1e5c(%rip),%xmm0        # 45a0 <__PRETTY_FUNCTION__.4+0x20>
    2743:	00 
    2744:	48 8b 74 24 08       	mov    0x8(%rsp),%rsi
    2749:	31 c0                	xor    %eax,%eax
    274b:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    2750:	b9 7e 00 00 00       	mov    $0x7e,%ecx
    2755:	ba eb 03 00 00       	mov    $0x3eb,%edx
    275a:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    275d:	48 8d 7c 24 25       	lea    0x25(%rsp),%rdi
    2762:	b8 20 00 00 00       	mov    $0x20,%eax
    2767:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
    276c:	c7 44 24 20 44 41 5d 	movl   $0x3a5d4144,0x20(%rsp)
    2773:	3a 
    2774:	66 89 44 24 24       	mov    %ax,0x24(%rsp)
    2779:	e8 32 fd ff ff       	call   24b0 <__strcpy_chk@plt>
    277e:	e8 5d fc ff ff       	call   23e0 <PyGILState_Ensure@plt>
    2783:	4c 89 e6             	mov    %r12,%rsi
    2786:	89 c5                	mov    %eax,%ebp
    2788:	48 8b 05 39 38 00 00 	mov    0x3839(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    278f:	48 8b 38             	mov    (%rax),%rdi
    2792:	e8 29 fc ff ff       	call   23c0 <PyErr_SetString@plt>
    2797:	89 ef                	mov    %ebp,%edi
    2799:	e8 c2 fb ff ff       	call   2360 <PyGILState_Release@plt>
    279e:	48 8b 84 24 18 04 00 	mov    0x418(%rsp),%rax
    27a5:	00 
    27a6:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    27ad:	00 00 
    27af:	75 0d                	jne    27be <gpuAssert.part.0+0xae>
    27b1:	48 81 c4 28 04 00 00 	add    $0x428,%rsp
    27b8:	31 c0                	xor    %eax,%eax
    27ba:	5d                   	pop    %rbp
    27bb:	41 5c                	pop    %r12
    27bd:	c3                   	ret    
    27be:	e8 ed fb ff ff       	call   23b0 <__stack_chk_fail@plt>
    27c3:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    27ca:	00 00 00 00 
    27ce:	66 90                	xchg   %ax,%ax

00000000000027d0 <setPrintfFifoSize>:
    27d0:	f3 0f 1e fa          	endbr64 
    27d4:	41 54                	push   %r12
    27d6:	48 89 f7             	mov    %rsi,%rdi
    27d9:	48 8d 35 27 1b 00 00 	lea    0x1b27(%rip),%rsi        # 4307 <_fini+0x927>
    27e0:	55                   	push   %rbp
    27e1:	48 83 ec 28          	sub    $0x28,%rsp
    27e5:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    27ec:	00 00 
    27ee:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    27f3:	31 c0                	xor    %eax,%eax
    27f5:	48 89 e2             	mov    %rsp,%rdx
    27f8:	e8 83 fc ff ff       	call   2480 <_PyArg_ParseTuple_SizeT@plt>
    27fd:	85 c0                	test   %eax,%eax
    27ff:	0f 84 c8 00 00 00    	je     28cd <setPrintfFifoSize+0xfd>
    2805:	48 83 3c 24 00       	cmpq   $0x0,(%rsp)
    280a:	0f 88 30 01 00 00    	js     2940 <setPrintfFifoSize+0x170>
    2810:	e8 ab fc ff ff       	call   24c0 <PyEval_SaveThread@plt>
    2815:	4c 8d 64 24 08       	lea    0x8(%rsp),%r12
    281a:	48 c7 44 24 08 00 00 	movq   $0x0,0x8(%rsp)
    2821:	00 00 
    2823:	4c 89 e7             	mov    %r12,%rdi
    2826:	48 89 c5             	mov    %rax,%rbp
    2829:	e8 e2 fc ff ff       	call   2510 <cuCtxGetCurrent@plt>
    282e:	89 c7                	mov    %eax,%edi
    2830:	85 c0                	test   %eax,%eax
    2832:	0f 85 80 00 00 00    	jne    28b8 <setPrintfFifoSize+0xe8>
    2838:	48 83 7c 24 08 00    	cmpq   $0x0,0x8(%rsp)
    283e:	0f 84 b4 00 00 00    	je     28f8 <setPrintfFifoSize+0x128>
    2844:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
    2849:	be 01 00 00 00       	mov    $0x1,%esi
    284e:	48 c7 44 24 10 00 00 	movq   $0x0,0x10(%rsp)
    2855:	00 00 
    2857:	e8 04 fd ff ff       	call   2560 <cuCtxGetLimit@plt>
    285c:	89 c7                	mov    %eax,%edi
    285e:	85 c0                	test   %eax,%eax
    2860:	75 76                	jne    28d8 <setPrintfFifoSize+0x108>
    2862:	48 8b 34 24          	mov    (%rsp),%rsi
    2866:	48 39 74 24 10       	cmp    %rsi,0x10(%rsp)
    286b:	74 19                	je     2886 <setPrintfFifoSize+0xb6>
    286d:	bf 01 00 00 00       	mov    $0x1,%edi
    2872:	e8 19 fc ff ff       	call   2490 <cuCtxSetLimit@plt>
    2877:	89 c7                	mov    %eax,%edi
    2879:	85 c0                	test   %eax,%eax
    287b:	74 09                	je     2886 <setPrintfFifoSize+0xb6>
    287d:	e8 8e fe ff ff       	call   2710 <gpuAssert.part.0>
    2882:	84 c0                	test   %al,%al
    2884:	74 3f                	je     28c5 <setPrintfFifoSize+0xf5>
    2886:	48 89 ef             	mov    %rbp,%rdi
    2889:	e8 e2 fa ff ff       	call   2370 <PyEval_RestoreThread@plt>
    288e:	48 8b 05 53 37 00 00 	mov    0x3753(%rip),%rax        # 5fe8 <_Py_NoneStruct@Base>
    2895:	48 83 00 01          	addq   $0x1,(%rax)
    2899:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
    289e:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    28a5:	00 00 
    28a7:	0f 85 b0 00 00 00    	jne    295d <setPrintfFifoSize+0x18d>
    28ad:	48 83 c4 28          	add    $0x28,%rsp
    28b1:	5d                   	pop    %rbp
    28b2:	41 5c                	pop    %r12
    28b4:	c3                   	ret    
    28b5:	0f 1f 00             	nopl   (%rax)
    28b8:	e8 53 fe ff ff       	call   2710 <gpuAssert.part.0>
    28bd:	84 c0                	test   %al,%al
    28bf:	0f 85 73 ff ff ff    	jne    2838 <setPrintfFifoSize+0x68>
    28c5:	48 89 ef             	mov    %rbp,%rdi
    28c8:	e8 a3 fa ff ff       	call   2370 <PyEval_RestoreThread@plt>
    28cd:	31 c0                	xor    %eax,%eax
    28cf:	eb c8                	jmp    2899 <setPrintfFifoSize+0xc9>
    28d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    28d8:	e8 33 fe ff ff       	call   2710 <gpuAssert.part.0>
    28dd:	84 c0                	test   %al,%al
    28df:	74 e4                	je     28c5 <setPrintfFifoSize+0xf5>
    28e1:	48 8b 34 24          	mov    (%rsp),%rsi
    28e5:	48 39 74 24 10       	cmp    %rsi,0x10(%rsp)
    28ea:	0f 85 7d ff ff ff    	jne    286d <setPrintfFifoSize+0x9d>
    28f0:	eb 94                	jmp    2886 <setPrintfFifoSize+0xb6>
    28f2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    28f8:	4c 89 e7             	mov    %r12,%rdi
    28fb:	31 f6                	xor    %esi,%esi
    28fd:	e8 2e fb ff ff       	call   2430 <cuDevicePrimaryCtxRetain@plt>
    2902:	89 c7                	mov    %eax,%edi
    2904:	85 c0                	test   %eax,%eax
    2906:	75 28                	jne    2930 <setPrintfFifoSize+0x160>
    2908:	48 8b 7c 24 08       	mov    0x8(%rsp),%rdi
    290d:	e8 3e fa ff ff       	call   2350 <cuCtxSetCurrent@plt>
    2912:	89 c7                	mov    %eax,%edi
    2914:	85 c0                	test   %eax,%eax
    2916:	0f 84 28 ff ff ff    	je     2844 <setPrintfFifoSize+0x74>
    291c:	e8 ef fd ff ff       	call   2710 <gpuAssert.part.0>
    2921:	84 c0                	test   %al,%al
    2923:	0f 85 1b ff ff ff    	jne    2844 <setPrintfFifoSize+0x74>
    2929:	eb 9a                	jmp    28c5 <setPrintfFifoSize+0xf5>
    292b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2930:	e8 db fd ff ff       	call   2710 <gpuAssert.part.0>
    2935:	84 c0                	test   %al,%al
    2937:	75 cf                	jne    2908 <setPrintfFifoSize+0x138>
    2939:	eb 8a                	jmp    28c5 <setPrintfFifoSize+0xf5>
    293b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2940:	48 8b 05 89 36 00 00 	mov    0x3689(%rip),%rax        # 5fd0 <PyExc_ValueError@Base>
    2947:	48 8d 35 ea 16 00 00 	lea    0x16ea(%rip),%rsi        # 4038 <_fini+0x658>
    294e:	48 8b 38             	mov    (%rax),%rdi
    2951:	e8 6a fa ff ff       	call   23c0 <PyErr_SetString@plt>
    2956:	31 c0                	xor    %eax,%eax
    2958:	e9 3c ff ff ff       	jmp    2899 <setPrintfFifoSize+0xc9>
    295d:	e8 4e fa ff ff       	call   23b0 <__stack_chk_fail@plt>
    2962:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2969:	00 00 00 00 
    296d:	0f 1f 00             	nopl   (%rax)

0000000000002970 <getDeviceProperties>:
    2970:	f3 0f 1e fa          	endbr64 
    2974:	48 83 ec 38          	sub    $0x38,%rsp
    2978:	48 89 f7             	mov    %rsi,%rdi
    297b:	48 8d 35 40 1a 00 00 	lea    0x1a40(%rip),%rsi        # 43c2 <_fini+0x9e2>
    2982:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2989:	00 00 
    298b:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    2990:	31 c0                	xor    %eax,%eax
    2992:	48 8d 54 24 04       	lea    0x4(%rsp),%rdx
    2997:	e8 e4 fa ff ff       	call   2480 <_PyArg_ParseTuple_SizeT@plt>
    299c:	85 c0                	test   %eax,%eax
    299e:	0f 84 7c 01 00 00    	je     2b20 <getDeviceProperties+0x1b0>
    29a4:	8b 74 24 04          	mov    0x4(%rsp),%esi
    29a8:	48 8d 7c 24 08       	lea    0x8(%rsp),%rdi
    29ad:	e8 3e fa ff ff       	call   23f0 <cuDeviceGet@plt>
    29b2:	8b 54 24 08          	mov    0x8(%rsp),%edx
    29b6:	48 8d 7c 24 0c       	lea    0xc(%rsp),%rdi
    29bb:	be 61 00 00 00       	mov    $0x61,%esi
    29c0:	e8 3b fb ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    29c5:	89 c7                	mov    %eax,%edi
    29c7:	85 c0                	test   %eax,%eax
    29c9:	0f 85 41 01 00 00    	jne    2b10 <getDeviceProperties+0x1a0>
    29cf:	8b 54 24 08          	mov    0x8(%rsp),%edx
    29d3:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
    29d8:	be 0c 00 00 00       	mov    $0xc,%esi
    29dd:	e8 1e fb ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    29e2:	89 c7                	mov    %eax,%edi
    29e4:	85 c0                	test   %eax,%eax
    29e6:	0f 85 54 01 00 00    	jne    2b40 <getDeviceProperties+0x1d0>
    29ec:	8b 54 24 08          	mov    0x8(%rsp),%edx
    29f0:	48 8d 7c 24 14       	lea    0x14(%rsp),%rdi
    29f5:	be 10 00 00 00       	mov    $0x10,%esi
    29fa:	e8 01 fb ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    29ff:	89 c7                	mov    %eax,%edi
    2a01:	85 c0                	test   %eax,%eax
    2a03:	0f 85 47 01 00 00    	jne    2b50 <getDeviceProperties+0x1e0>
    2a09:	8b 54 24 08          	mov    0x8(%rsp),%edx
    2a0d:	48 8d 7c 24 18       	lea    0x18(%rsp),%rdi
    2a12:	be 0a 00 00 00       	mov    $0xa,%esi
    2a17:	e8 e4 fa ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2a1c:	89 c7                	mov    %eax,%edi
    2a1e:	85 c0                	test   %eax,%eax
    2a20:	0f 85 3a 01 00 00    	jne    2b60 <getDeviceProperties+0x1f0>
    2a26:	8b 54 24 08          	mov    0x8(%rsp),%edx
    2a2a:	48 8d 7c 24 1c       	lea    0x1c(%rsp),%rdi
    2a2f:	be 0d 00 00 00       	mov    $0xd,%esi
    2a34:	e8 c7 fa ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2a39:	89 c7                	mov    %eax,%edi
    2a3b:	85 c0                	test   %eax,%eax
    2a3d:	74 0d                	je     2a4c <getDeviceProperties+0xdc>
    2a3f:	e8 cc fc ff ff       	call   2710 <gpuAssert.part.0>
    2a44:	84 c0                	test   %al,%al
    2a46:	0f 84 d4 00 00 00    	je     2b20 <getDeviceProperties+0x1b0>
    2a4c:	8b 54 24 08          	mov    0x8(%rsp),%edx
    2a50:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    2a55:	be 24 00 00 00       	mov    $0x24,%esi
    2a5a:	e8 a1 fa ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2a5f:	89 c7                	mov    %eax,%edi
    2a61:	85 c0                	test   %eax,%eax
    2a63:	74 0d                	je     2a72 <getDeviceProperties+0x102>
    2a65:	e8 a6 fc ff ff       	call   2710 <gpuAssert.part.0>
    2a6a:	84 c0                	test   %al,%al
    2a6c:	0f 84 ae 00 00 00    	je     2b20 <getDeviceProperties+0x1b0>
    2a72:	8b 54 24 08          	mov    0x8(%rsp),%edx
    2a76:	48 8d 7c 24 24       	lea    0x24(%rsp),%rdi
    2a7b:	be 25 00 00 00       	mov    $0x25,%esi
    2a80:	e8 7b fa ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2a85:	89 c7                	mov    %eax,%edi
    2a87:	85 c0                	test   %eax,%eax
    2a89:	74 0d                	je     2a98 <getDeviceProperties+0x128>
    2a8b:	e8 80 fc ff ff       	call   2710 <gpuAssert.part.0>
    2a90:	84 c0                	test   %al,%al
    2a92:	0f 84 88 00 00 00    	je     2b20 <getDeviceProperties+0x1b0>
    2a98:	48 83 ec 08          	sub    $0x8,%rsp
    2a9c:	4c 8d 0d 66 18 00 00 	lea    0x1866(%rip),%r9        # 4309 <_fini+0x929>
    2aa3:	48 8d 0d 74 18 00 00 	lea    0x1874(%rip),%rcx        # 431e <_fini+0x93e>
    2aaa:	8b 44 24 2c          	mov    0x2c(%rsp),%eax
    2aae:	48 8d 35 76 18 00 00 	lea    0x1876(%rip),%rsi        # 432b <_fini+0x94b>
    2ab5:	48 8d 3d 9c 15 00 00 	lea    0x159c(%rip),%rdi        # 4058 <_fini+0x678>
    2abc:	50                   	push   %rax
    2abd:	48 8d 05 76 18 00 00 	lea    0x1876(%rip),%rax        # 433a <_fini+0x95a>
    2ac4:	50                   	push   %rax
    2ac5:	8b 44 24 38          	mov    0x38(%rsp),%eax
    2ac9:	50                   	push   %rax
    2aca:	48 8d 05 77 18 00 00 	lea    0x1877(%rip),%rax        # 4348 <_fini+0x968>
    2ad1:	50                   	push   %rax
    2ad2:	8b 44 24 44          	mov    0x44(%rsp),%eax
    2ad6:	50                   	push   %rax
    2ad7:	48 8d 05 79 18 00 00 	lea    0x1879(%rip),%rax        # 4357 <_fini+0x977>
    2ade:	50                   	push   %rax
    2adf:	8b 44 24 50          	mov    0x50(%rsp),%eax
    2ae3:	50                   	push   %rax
    2ae4:	48 8d 05 7a 18 00 00 	lea    0x187a(%rip),%rax        # 4365 <_fini+0x985>
    2aeb:	50                   	push   %rax
    2aec:	8b 44 24 5c          	mov    0x5c(%rsp),%eax
    2af0:	50                   	push   %rax
    2af1:	44 8b 44 24 60       	mov    0x60(%rsp),%r8d
    2af6:	31 c0                	xor    %eax,%eax
    2af8:	8b 54 24 5c          	mov    0x5c(%rsp),%edx
    2afc:	e8 1f fa ff ff       	call   2520 <_Py_BuildValue_SizeT@plt>
    2b01:	48 83 c4 50          	add    $0x50,%rsp
    2b05:	eb 1b                	jmp    2b22 <getDeviceProperties+0x1b2>
    2b07:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2b0e:	00 00 
    2b10:	e8 fb fb ff ff       	call   2710 <gpuAssert.part.0>
    2b15:	84 c0                	test   %al,%al
    2b17:	0f 85 b2 fe ff ff    	jne    29cf <getDeviceProperties+0x5f>
    2b1d:	0f 1f 00             	nopl   (%rax)
    2b20:	31 c0                	xor    %eax,%eax
    2b22:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    2b27:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    2b2e:	00 00 
    2b30:	75 3d                	jne    2b6f <getDeviceProperties+0x1ff>
    2b32:	48 83 c4 38          	add    $0x38,%rsp
    2b36:	c3                   	ret    
    2b37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2b3e:	00 00 
    2b40:	e8 cb fb ff ff       	call   2710 <gpuAssert.part.0>
    2b45:	84 c0                	test   %al,%al
    2b47:	0f 85 9f fe ff ff    	jne    29ec <getDeviceProperties+0x7c>
    2b4d:	eb d1                	jmp    2b20 <getDeviceProperties+0x1b0>
    2b4f:	90                   	nop
    2b50:	e8 bb fb ff ff       	call   2710 <gpuAssert.part.0>
    2b55:	84 c0                	test   %al,%al
    2b57:	0f 85 ac fe ff ff    	jne    2a09 <getDeviceProperties+0x99>
    2b5d:	eb c1                	jmp    2b20 <getDeviceProperties+0x1b0>
    2b5f:	90                   	nop
    2b60:	e8 ab fb ff ff       	call   2710 <gpuAssert.part.0>
    2b65:	84 c0                	test   %al,%al
    2b67:	0f 85 b9 fe ff ff    	jne    2a26 <getDeviceProperties+0xb6>
    2b6d:	eb b1                	jmp    2b20 <getDeviceProperties+0x1b0>
    2b6f:	e8 3c f8 ff ff       	call   23b0 <__stack_chk_fail@plt>
    2b74:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    2b7b:	00 00 00 00 
    2b7f:	90                   	nop

0000000000002b80 <loadBinary>:
    2b80:	f3 0f 1e fa          	endbr64 
    2b84:	41 54                	push   %r12
    2b86:	48 89 f7             	mov    %rsi,%rdi
    2b89:	48 8d 35 de 17 00 00 	lea    0x17de(%rip),%rsi        # 436e <_fini+0x98e>
    2b90:	55                   	push   %rbp
    2b91:	48 83 ec 68          	sub    $0x68,%rsp
    2b95:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2b9c:	00 00 
    2b9e:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    2ba3:	31 c0                	xor    %eax,%eax
    2ba5:	48 8d 4c 24 30       	lea    0x30(%rsp),%rcx
    2baa:	48 8d 54 24 28       	lea    0x28(%rsp),%rdx
    2baf:	48 83 ec 08          	sub    $0x8,%rsp
    2bb3:	48 8d 44 24 14       	lea    0x14(%rsp),%rax
    2bb8:	50                   	push   %rax
    2bb9:	31 c0                	xor    %eax,%eax
    2bbb:	4c 8d 4c 24 18       	lea    0x18(%rsp),%r9
    2bc0:	4c 8d 44 24 48       	lea    0x48(%rsp),%r8
    2bc5:	e8 b6 f8 ff ff       	call   2480 <_PyArg_ParseTuple_SizeT@plt>
    2bca:	5a                   	pop    %rdx
    2bcb:	59                   	pop    %rcx
    2bcc:	85 c0                	test   %eax,%eax
    2bce:	0f 84 74 01 00 00    	je     2d48 <loadBinary+0x1c8>
    2bd4:	4c 8d 64 24 50       	lea    0x50(%rsp),%r12
    2bd9:	c7 44 24 10 00 00 00 	movl   $0x0,0x10(%rsp)
    2be0:	00 
    2be1:	c7 44 24 14 00 00 00 	movl   $0x0,0x14(%rsp)
    2be8:	00 
    2be9:	c7 44 24 18 00 00 00 	movl   $0x0,0x18(%rsp)
    2bf0:	00 
    2bf1:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
    2bf8:	00 00 
    2bfa:	e8 c1 f8 ff ff       	call   24c0 <PyEval_SaveThread@plt>
    2bff:	4c 89 e7             	mov    %r12,%rdi
    2c02:	48 89 c5             	mov    %rax,%rbp
    2c05:	e8 06 f9 ff ff       	call   2510 <cuCtxGetCurrent@plt>
    2c0a:	89 c7                	mov    %eax,%edi
    2c0c:	85 c0                	test   %eax,%eax
    2c0e:	0f 85 1c 01 00 00    	jne    2d30 <loadBinary+0x1b0>
    2c14:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
    2c1a:	0f 84 b8 01 00 00    	je     2dd8 <loadBinary+0x258>
    2c20:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    2c25:	48 8d 7c 24 48       	lea    0x48(%rsp),%rdi
    2c2a:	e8 e1 f6 ff ff       	call   2310 <cuModuleLoadData@plt>
    2c2f:	89 c7                	mov    %eax,%edi
    2c31:	85 c0                	test   %eax,%eax
    2c33:	0f 85 37 01 00 00    	jne    2d70 <loadBinary+0x1f0>
    2c39:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
    2c3e:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    2c43:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    2c48:	e8 93 f6 ff ff       	call   22e0 <cuModuleGetFunction@plt>
    2c4d:	89 c7                	mov    %eax,%edi
    2c4f:	85 c0                	test   %eax,%eax
    2c51:	0f 85 29 01 00 00    	jne    2d80 <loadBinary+0x200>
    2c57:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2c5c:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi
    2c61:	be 04 00 00 00       	mov    $0x4,%esi
    2c66:	e8 d5 f6 ff ff       	call   2340 <cuFuncGetAttribute@plt>
    2c6b:	89 c7                	mov    %eax,%edi
    2c6d:	85 c0                	test   %eax,%eax
    2c6f:	0f 85 1b 01 00 00    	jne    2d90 <loadBinary+0x210>
    2c75:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2c7a:	48 8d 7c 24 14       	lea    0x14(%rsp),%rdi
    2c7f:	be 03 00 00 00       	mov    $0x3,%esi
    2c84:	e8 b7 f6 ff ff       	call   2340 <cuFuncGetAttribute@plt>
    2c89:	89 c7                	mov    %eax,%edi
    2c8b:	85 c0                	test   %eax,%eax
    2c8d:	0f 85 0d 01 00 00    	jne    2da0 <loadBinary+0x220>
    2c93:	8b 54 24 14          	mov    0x14(%rsp),%edx
    2c97:	48 8d 7c 24 18       	lea    0x18(%rsp),%rdi
    2c9c:	85 d2                	test   %edx,%edx
    2c9e:	8d 42 03             	lea    0x3(%rdx),%eax
    2ca1:	0f 49 c2             	cmovns %edx,%eax
    2ca4:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2ca9:	31 f6                	xor    %esi,%esi
    2cab:	c1 f8 02             	sar    $0x2,%eax
    2cae:	89 44 24 14          	mov    %eax,0x14(%rsp)
    2cb2:	e8 89 f6 ff ff       	call   2340 <cuFuncGetAttribute@plt>
    2cb7:	89 c7                	mov    %eax,%edi
    2cb9:	85 c0                	test   %eax,%eax
    2cbb:	0f 85 ef 00 00 00    	jne    2db0 <loadBinary+0x230>
    2cc1:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    2cc5:	48 8d 7c 24 1c       	lea    0x1c(%rsp),%rdi
    2cca:	be 61 00 00 00       	mov    $0x61,%esi
    2ccf:	e8 2c f8 ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2cd4:	89 c7                	mov    %eax,%edi
    2cd6:	85 c0                	test   %eax,%eax
    2cd8:	0f 85 e2 00 00 00    	jne    2dc0 <loadBinary+0x240>
    2cde:	81 7c 24 08 00 c0 00 	cmpl   $0xc000,0x8(%rsp)
    2ce5:	00 
    2ce6:	7e 0e                	jle    2cf6 <loadBinary+0x176>
    2ce8:	81 7c 24 1c 00 c0 00 	cmpl   $0xc000,0x1c(%rsp)
    2cef:	00 
    2cf0:	0f 8f 2a 01 00 00    	jg     2e20 <loadBinary+0x2a0>
    2cf6:	48 89 ef             	mov    %rbp,%rdi
    2cf9:	e8 72 f6 ff ff       	call   2370 <PyEval_RestoreThread@plt>
    2cfe:	e8 3d f7 ff ff       	call   2440 <PyErr_Occurred@plt>
    2d03:	48 85 c0             	test   %rax,%rax
    2d06:	75 40                	jne    2d48 <loadBinary+0x1c8>
    2d08:	44 8b 4c 24 18       	mov    0x18(%rsp),%r9d
    2d0d:	44 8b 44 24 14       	mov    0x14(%rsp),%r8d
    2d12:	31 c0                	xor    %eax,%eax
    2d14:	48 8d 3d 59 16 00 00 	lea    0x1659(%rip),%rdi        # 4374 <_fini+0x994>
    2d1b:	8b 4c 24 10          	mov    0x10(%rsp),%ecx
    2d1f:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2d24:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
    2d29:	e8 f2 f7 ff ff       	call   2520 <_Py_BuildValue_SizeT@plt>
    2d2e:	eb 1a                	jmp    2d4a <loadBinary+0x1ca>
    2d30:	e8 db f9 ff ff       	call   2710 <gpuAssert.part.0>
    2d35:	84 c0                	test   %al,%al
    2d37:	0f 85 d7 fe ff ff    	jne    2c14 <loadBinary+0x94>
    2d3d:	0f 1f 00             	nopl   (%rax)
    2d40:	48 89 ef             	mov    %rbp,%rdi
    2d43:	e8 28 f6 ff ff       	call   2370 <PyEval_RestoreThread@plt>
    2d48:	31 c0                	xor    %eax,%eax
    2d4a:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
    2d4f:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    2d56:	00 00 
    2d58:	0f 85 68 01 00 00    	jne    2ec6 <loadBinary+0x346>
    2d5e:	48 83 c4 68          	add    $0x68,%rsp
    2d62:	5d                   	pop    %rbp
    2d63:	41 5c                	pop    %r12
    2d65:	c3                   	ret    
    2d66:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2d6d:	00 00 00 
    2d70:	e8 9b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2d75:	84 c0                	test   %al,%al
    2d77:	0f 85 bc fe ff ff    	jne    2c39 <loadBinary+0xb9>
    2d7d:	eb c1                	jmp    2d40 <loadBinary+0x1c0>
    2d7f:	90                   	nop
    2d80:	e8 8b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2d85:	84 c0                	test   %al,%al
    2d87:	0f 85 ca fe ff ff    	jne    2c57 <loadBinary+0xd7>
    2d8d:	eb b1                	jmp    2d40 <loadBinary+0x1c0>
    2d8f:	90                   	nop
    2d90:	e8 7b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2d95:	84 c0                	test   %al,%al
    2d97:	0f 85 d8 fe ff ff    	jne    2c75 <loadBinary+0xf5>
    2d9d:	eb a1                	jmp    2d40 <loadBinary+0x1c0>
    2d9f:	90                   	nop
    2da0:	e8 6b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2da5:	84 c0                	test   %al,%al
    2da7:	0f 85 e6 fe ff ff    	jne    2c93 <loadBinary+0x113>
    2dad:	eb 91                	jmp    2d40 <loadBinary+0x1c0>
    2daf:	90                   	nop
    2db0:	e8 5b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2db5:	84 c0                	test   %al,%al
    2db7:	0f 85 04 ff ff ff    	jne    2cc1 <loadBinary+0x141>
    2dbd:	eb 81                	jmp    2d40 <loadBinary+0x1c0>
    2dbf:	90                   	nop
    2dc0:	e8 4b f9 ff ff       	call   2710 <gpuAssert.part.0>
    2dc5:	84 c0                	test   %al,%al
    2dc7:	0f 85 11 ff ff ff    	jne    2cde <loadBinary+0x15e>
    2dcd:	e9 6e ff ff ff       	jmp    2d40 <loadBinary+0x1c0>
    2dd2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2dd8:	8b 74 24 0c          	mov    0xc(%rsp),%esi
    2ddc:	4c 89 e7             	mov    %r12,%rdi
    2ddf:	e8 4c f6 ff ff       	call   2430 <cuDevicePrimaryCtxRetain@plt>
    2de4:	89 c7                	mov    %eax,%edi
    2de6:	85 c0                	test   %eax,%eax
    2de8:	75 26                	jne    2e10 <loadBinary+0x290>
    2dea:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    2def:	e8 5c f5 ff ff       	call   2350 <cuCtxSetCurrent@plt>
    2df4:	89 c7                	mov    %eax,%edi
    2df6:	85 c0                	test   %eax,%eax
    2df8:	0f 84 22 fe ff ff    	je     2c20 <loadBinary+0xa0>
    2dfe:	e8 0d f9 ff ff       	call   2710 <gpuAssert.part.0>
    2e03:	84 c0                	test   %al,%al
    2e05:	0f 85 15 fe ff ff    	jne    2c20 <loadBinary+0xa0>
    2e0b:	e9 30 ff ff ff       	jmp    2d40 <loadBinary+0x1c0>
    2e10:	e8 fb f8 ff ff       	call   2710 <gpuAssert.part.0>
    2e15:	84 c0                	test   %al,%al
    2e17:	75 d1                	jne    2dea <loadBinary+0x26a>
    2e19:	e9 22 ff ff ff       	jmp    2d40 <loadBinary+0x1c0>
    2e1e:	66 90                	xchg   %ax,%ax
    2e20:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    2e25:	be 01 00 00 00       	mov    $0x1,%esi
    2e2a:	e8 11 f7 ff ff       	call   2540 <cuFuncSetCacheConfig@plt>
    2e2f:	89 c7                	mov    %eax,%edi
    2e31:	85 c0                	test   %eax,%eax
    2e33:	74 0d                	je     2e42 <loadBinary+0x2c2>
    2e35:	e8 d6 f8 ff ff       	call   2710 <gpuAssert.part.0>
    2e3a:	84 c0                	test   %al,%al
    2e3c:	0f 84 fe fe ff ff    	je     2d40 <loadBinary+0x1c0>
    2e42:	8b 54 24 0c          	mov    0xc(%rsp),%edx
    2e46:	48 8d 7c 24 20       	lea    0x20(%rsp),%rdi
    2e4b:	be 51 00 00 00       	mov    $0x51,%esi
    2e50:	e8 ab f6 ff ff       	call   2500 <cuDeviceGetAttribute@plt>
    2e55:	89 c7                	mov    %eax,%edi
    2e57:	85 c0                	test   %eax,%eax
    2e59:	74 0d                	je     2e68 <loadBinary+0x2e8>
    2e5b:	e8 b0 f8 ff ff       	call   2710 <gpuAssert.part.0>
    2e60:	84 c0                	test   %al,%al
    2e62:	0f 84 d8 fe ff ff    	je     2d40 <loadBinary+0x1c0>
    2e68:	48 8b 54 24 40       	mov    0x40(%rsp),%rdx
    2e6d:	48 8d 7c 24 24       	lea    0x24(%rsp),%rdi
    2e72:	be 01 00 00 00       	mov    $0x1,%esi
    2e77:	e8 c4 f4 ff ff       	call   2340 <cuFuncGetAttribute@plt>
    2e7c:	89 c7                	mov    %eax,%edi
    2e7e:	85 c0                	test   %eax,%eax
    2e80:	75 36                	jne    2eb8 <loadBinary+0x338>
    2e82:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    2e87:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
    2e8b:	be 08 00 00 00       	mov    $0x8,%esi
    2e90:	2b 54 24 24          	sub    0x24(%rsp),%edx
    2e94:	e8 57 f4 ff ff       	call   22f0 <cuFuncSetAttribute@plt>
    2e99:	89 c7                	mov    %eax,%edi
    2e9b:	85 c0                	test   %eax,%eax
    2e9d:	0f 84 53 fe ff ff    	je     2cf6 <loadBinary+0x176>
    2ea3:	e8 68 f8 ff ff       	call   2710 <gpuAssert.part.0>
    2ea8:	84 c0                	test   %al,%al
    2eaa:	0f 85 46 fe ff ff    	jne    2cf6 <loadBinary+0x176>
    2eb0:	e9 8b fe ff ff       	jmp    2d40 <loadBinary+0x1c0>
    2eb5:	0f 1f 00             	nopl   (%rax)
    2eb8:	e8 53 f8 ff ff       	call   2710 <gpuAssert.part.0>
    2ebd:	84 c0                	test   %al,%al
    2ebf:	75 c1                	jne    2e82 <loadBinary+0x302>
    2ec1:	e9 7a fe ff ff       	jmp    2d40 <loadBinary+0x1c0>
    2ec6:	e8 e5 f4 ff ff       	call   23b0 <__stack_chk_fail@plt>
    2ecb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002ed0 <occupancyMaxActiveClusters>:
    2ed0:	f3 0f 1e fa          	endbr64 
    2ed4:	41 55                	push   %r13
    2ed6:	48 89 f7             	mov    %rsi,%rdi
    2ed9:	48 8d 35 9c 14 00 00 	lea    0x149c(%rip),%rsi        # 437c <_fini+0x99c>
    2ee0:	41 54                	push   %r12
    2ee2:	55                   	push   %rbp
    2ee3:	48 81 ec d0 04 00 00 	sub    $0x4d0,%rsp
    2eea:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    2ef1:	00 00 
    2ef3:	48 89 84 24 c8 04 00 	mov    %rax,0x4c8(%rsp)
    2efa:	00 
    2efb:	31 c0                	xor    %eax,%eax
    2efd:	48 8d 4c 24 1c       	lea    0x1c(%rsp),%rcx
    2f02:	48 8d 54 24 20       	lea    0x20(%rsp),%rdx
    2f07:	c7 44 24 0c ff ff ff 	movl   $0xffffffff,0xc(%rsp)
    2f0e:	ff 
    2f0f:	c7 44 24 10 ff ff ff 	movl   $0xffffffff,0x10(%rsp)
    2f16:	ff 
    2f17:	48 83 ec 08          	sub    $0x8,%rsp
    2f1b:	c7 44 24 1c ff ff ff 	movl   $0xffffffff,0x1c(%rsp)
    2f22:	ff 
    2f23:	c7 44 24 20 ff ff ff 	movl   $0xffffffff,0x20(%rsp)
    2f2a:	ff 
    2f2b:	c7 44 24 24 00 00 00 	movl   $0x0,0x24(%rsp)
    2f32:	00 
    2f33:	48 8d 44 24 1c       	lea    0x1c(%rsp),%rax
    2f38:	50                   	push   %rax
    2f39:	31 c0                	xor    %eax,%eax
    2f3b:	4c 8d 4c 24 20       	lea    0x20(%rsp),%r9
    2f40:	4c 8d 44 24 1c       	lea    0x1c(%rsp),%r8
    2f45:	e8 36 f5 ff ff       	call   2480 <_PyArg_ParseTuple_SizeT@plt>
    2f4a:	59                   	pop    %rcx
    2f4b:	5e                   	pop    %rsi
    2f4c:	85 c0                	test   %eax,%eax
    2f4e:	0f 84 87 01 00 00    	je     30db <occupancyMaxActiveClusters+0x20b>
    2f54:	e8 67 f5 ff ff       	call   24c0 <PyEval_SaveThread@plt>
    2f59:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    2f5e:	8b 54 24 1c          	mov    0x1c(%rsp),%edx
    2f62:	be 08 00 00 00       	mov    $0x8,%esi
    2f67:	49 89 c4             	mov    %rax,%r12
    2f6a:	e8 81 f3 ff ff       	call   22f0 <cuFuncSetAttribute@plt>
    2f6f:	89 c7                	mov    %eax,%edi
    2f71:	85 c0                	test   %eax,%eax
    2f73:	0f 85 8f 01 00 00    	jne    3108 <occupancyMaxActiveClusters+0x238>
    2f79:	4c 89 e7             	mov    %r12,%rdi
    2f7c:	e8 ef f3 ff ff       	call   2370 <PyEval_RestoreThread@plt>
    2f81:	8b 44 24 14          	mov    0x14(%rsp),%eax
    2f85:	66 0f 6e 44 24 0c    	movd   0xc(%rsp),%xmm0
    2f8b:	c7 44 24 70 04 00 00 	movl   $0x4,0x70(%rsp)
    2f92:	00 
    2f93:	66 0f 6e 4c 24 10    	movd   0x10(%rsp),%xmm1
    2f99:	48 83 3d df 34 00 00 	cmpq   $0x0,0x34df(%rip)        # 6480 <cuOccupancyMaxActiveClusters.0>
    2fa0:	00 
    2fa1:	c7 44 24 44 01 00 00 	movl   $0x1,0x44(%rsp)
    2fa8:	00 
    2fa9:	89 84 24 80 00 00 00 	mov    %eax,0x80(%rsp)
    2fb0:	89 44 24 38          	mov    %eax,0x38(%rsp)
    2fb4:	48 8b 05 05 16 00 00 	mov    0x1605(%rip),%rax        # 45c0 <__PRETTY_FUNCTION__.4+0x40>
    2fbb:	66 0f 62 c1          	punpckldq %xmm1,%xmm0
    2fbf:	48 c7 44 24 50 00 00 	movq   $0x0,0x50(%rsp)
    2fc6:	00 00 
    2fc8:	48 89 44 24 3c       	mov    %rax,0x3c(%rsp)
    2fcd:	8b 44 24 1c          	mov    0x1c(%rsp),%eax
    2fd1:	c7 44 24 60 01 00 00 	movl   $0x1,0x60(%rsp)
    2fd8:	00 
    2fd9:	89 44 24 48          	mov    %eax,0x48(%rsp)
    2fdd:	48 8d 44 24 70       	lea    0x70(%rsp),%rax
    2fe2:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    2fe7:	66 0f d6 44 24 78    	movq   %xmm0,0x78(%rsp)
    2fed:	66 0f d6 44 24 30    	movq   %xmm0,0x30(%rsp)
    2ff3:	0f 84 77 01 00 00    	je     3170 <occupancyMaxActiveClusters+0x2a0>
    2ff9:	e8 c2 f4 ff ff       	call   24c0 <PyEval_SaveThread@plt>
    2ffe:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    3003:	ba 01 00 00 00       	mov    $0x1,%edx
    3008:	be 0e 00 00 00       	mov    $0xe,%esi
    300d:	49 89 c4             	mov    %rax,%r12
    3010:	e8 db f2 ff ff       	call   22f0 <cuFuncSetAttribute@plt>
    3015:	89 c7                	mov    %eax,%edi
    3017:	85 c0                	test   %eax,%eax
    3019:	75 35                	jne    3050 <occupancyMaxActiveClusters+0x180>
    301b:	48 8d 7c 24 18       	lea    0x18(%rsp),%rdi
    3020:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    3025:	48 8d 54 24 30       	lea    0x30(%rsp),%rdx
    302a:	ff 15 50 34 00 00    	call   *0x3450(%rip)        # 6480 <cuOccupancyMaxActiveClusters.0>
    3030:	89 c7                	mov    %eax,%edi
    3032:	85 c0                	test   %eax,%eax
    3034:	75 1a                	jne    3050 <occupancyMaxActiveClusters+0x180>
    3036:	4c 89 e7             	mov    %r12,%rdi
    3039:	e8 32 f3 ff ff       	call   2370 <PyEval_RestoreThread@plt>
    303e:	48 63 7c 24 18       	movslq 0x18(%rsp),%rdi
    3043:	e8 c8 f3 ff ff       	call   2410 <PyLong_FromLong@plt>
    3048:	49 89 c4             	mov    %rax,%r12
    304b:	e9 8e 00 00 00       	jmp    30de <occupancyMaxActiveClusters+0x20e>
    3050:	48 8d 74 24 28       	lea    0x28(%rsp),%rsi
    3055:	4c 8d ac 24 c0 00 00 	lea    0xc0(%rsp),%r13
    305c:	00 
    305d:	e8 3e f4 ff ff       	call   24a0 <cuGetErrorString@plt>
    3062:	31 c0                	xor    %eax,%eax
    3064:	48 8d bc 24 d0 00 00 	lea    0xd0(%rsp),%rdi
    306b:	00 
    306c:	b9 7e 00 00 00       	mov    $0x7e,%ecx
    3071:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    3074:	b8 20 00 00 00       	mov    $0x20,%eax
    3079:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    307e:	66 0f 6f 05 1a 15 00 	movdqa 0x151a(%rip),%xmm0        # 45a0 <__PRETTY_FUNCTION__.4+0x20>
    3085:	00 
    3086:	c7 84 24 d0 00 00 00 	movl   $0x3a5d4144,0xd0(%rsp)
    308d:	44 41 5d 3a 
    3091:	48 8d bc 24 d5 00 00 	lea    0xd5(%rsp),%rdi
    3098:	00 
    3099:	ba eb 03 00 00       	mov    $0x3eb,%edx
    309e:	66 89 84 24 d4 00 00 	mov    %ax,0xd4(%rsp)
    30a5:	00 
    30a6:	0f 29 84 24 c0 00 00 	movaps %xmm0,0xc0(%rsp)
    30ad:	00 
    30ae:	e8 fd f3 ff ff       	call   24b0 <__strcpy_chk@plt>
    30b3:	e8 28 f3 ff ff       	call   23e0 <PyGILState_Ensure@plt>
    30b8:	4c 89 ee             	mov    %r13,%rsi
    30bb:	89 c5                	mov    %eax,%ebp
    30bd:	48 8b 05 04 2f 00 00 	mov    0x2f04(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    30c4:	48 8b 38             	mov    (%rax),%rdi
    30c7:	e8 f4 f2 ff ff       	call   23c0 <PyErr_SetString@plt>
    30cc:	89 ef                	mov    %ebp,%edi
    30ce:	e8 8d f2 ff ff       	call   2360 <PyGILState_Release@plt>
    30d3:	4c 89 e7             	mov    %r12,%rdi
    30d6:	e8 95 f2 ff ff       	call   2370 <PyEval_RestoreThread@plt>
    30db:	45 31 e4             	xor    %r12d,%r12d
    30de:	48 8b 84 24 c8 04 00 	mov    0x4c8(%rsp),%rax
    30e5:	00 
    30e6:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    30ed:	00 00 
    30ef:	0f 85 2c 01 00 00    	jne    3221 <occupancyMaxActiveClusters+0x351>
    30f5:	48 81 c4 d0 04 00 00 	add    $0x4d0,%rsp
    30fc:	4c 89 e0             	mov    %r12,%rax
    30ff:	5d                   	pop    %rbp
    3100:	41 5c                	pop    %r12
    3102:	41 5d                	pop    %r13
    3104:	c3                   	ret    
    3105:	0f 1f 00             	nopl   (%rax)
    3108:	48 8d 74 24 30       	lea    0x30(%rsp),%rsi
    310d:	4c 8d ac 24 c0 00 00 	lea    0xc0(%rsp),%r13
    3114:	00 
    3115:	e8 86 f3 ff ff       	call   24a0 <cuGetErrorString@plt>
    311a:	66 0f 6f 05 7e 14 00 	movdqa 0x147e(%rip),%xmm0        # 45a0 <__PRETTY_FUNCTION__.4+0x20>
    3121:	00 
    3122:	ba 20 00 00 00       	mov    $0x20,%edx
    3127:	48 8d bc 24 d0 00 00 	lea    0xd0(%rsp),%rdi
    312e:	00 
    312f:	b9 7e 00 00 00       	mov    $0x7e,%ecx
    3134:	31 c0                	xor    %eax,%eax
    3136:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
    313b:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    313e:	66 89 94 24 d4 00 00 	mov    %dx,0xd4(%rsp)
    3145:	00 
    3146:	ba eb 03 00 00       	mov    $0x3eb,%edx
    314b:	48 8d bc 24 d5 00 00 	lea    0xd5(%rsp),%rdi
    3152:	00 
    3153:	c7 84 24 d0 00 00 00 	movl   $0x3a5d4144,0xd0(%rsp)
    315a:	44 41 5d 3a 
    315e:	0f 29 84 24 c0 00 00 	movaps %xmm0,0xc0(%rsp)
    3165:	00 
    3166:	e9 43 ff ff ff       	jmp    30ae <occupancyMaxActiveClusters+0x1de>
    316b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    3170:	be 01 00 00 00       	mov    $0x1,%esi
    3175:	48 8d 3d 15 12 00 00 	lea    0x1215(%rip),%rdi        # 4391 <_fini+0x9b1>
    317c:	e8 9f f2 ff ff       	call   2420 <dlopen@plt>
    3181:	49 89 c4             	mov    %rax,%r12
    3184:	48 85 c0             	test   %rax,%rax
    3187:	74 37                	je     31c0 <occupancyMaxActiveClusters+0x2f0>
    3189:	e8 92 f1 ff ff       	call   2320 <dlerror@plt>
    318e:	48 8d 35 09 12 00 00 	lea    0x1209(%rip),%rsi        # 439e <_fini+0x9be>
    3195:	4c 89 e7             	mov    %r12,%rdi
    3198:	e8 43 f3 ff ff       	call   24e0 <dlsym@plt>
    319d:	48 89 c5             	mov    %rax,%rbp
    31a0:	e8 7b f1 ff ff       	call   2320 <dlerror@plt>
    31a5:	48 85 c0             	test   %rax,%rax
    31a8:	75 46                	jne    31f0 <occupancyMaxActiveClusters+0x320>
    31aa:	48 89 2d cf 32 00 00 	mov    %rbp,0x32cf(%rip)        # 6480 <cuOccupancyMaxActiveClusters.0>
    31b1:	48 85 ed             	test   %rbp,%rbp
    31b4:	0f 85 3f fe ff ff    	jne    2ff9 <occupancyMaxActiveClusters+0x129>
    31ba:	e9 1c ff ff ff       	jmp    30db <occupancyMaxActiveClusters+0x20b>
    31bf:	90                   	nop
    31c0:	48 8b 05 01 2e 00 00 	mov    0x2e01(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    31c7:	48 8d 35 b4 11 00 00 	lea    0x11b4(%rip),%rsi        # 4382 <_fini+0x9a2>
    31ce:	48 8b 38             	mov    (%rax),%rdi
    31d1:	e8 ea f1 ff ff       	call   23c0 <PyErr_SetString@plt>
    31d6:	48 c7 05 9f 32 00 00 	movq   $0x0,0x329f(%rip)        # 6480 <cuOccupancyMaxActiveClusters.0>
    31dd:	00 00 00 00 
    31e1:	e9 f8 fe ff ff       	jmp    30de <occupancyMaxActiveClusters+0x20e>
    31e6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    31ed:	00 00 00 
    31f0:	48 8b 05 d1 2d 00 00 	mov    0x2dd1(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    31f7:	48 8d 35 82 0e 00 00 	lea    0xe82(%rip),%rsi        # 4080 <_fini+0x6a0>
    31fe:	48 8b 38             	mov    (%rax),%rdi
    3201:	e8 ba f1 ff ff       	call   23c0 <PyErr_SetString@plt>
    3206:	4c 89 e7             	mov    %r12,%rdi
    3209:	45 31 e4             	xor    %r12d,%r12d
    320c:	e8 3f f3 ff ff       	call   2550 <dlclose@plt>
    3211:	48 c7 05 64 32 00 00 	movq   $0x0,0x3264(%rip)        # 6480 <cuOccupancyMaxActiveClusters.0>
    3218:	00 00 00 00 
    321c:	e9 bd fe ff ff       	jmp    30de <occupancyMaxActiveClusters+0x20e>
    3221:	e8 8a f1 ff ff       	call   23b0 <__stack_chk_fail@plt>
    3226:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    322d:	00 00 00 

0000000000003230 <fillTMADescriptor>:
    3230:	f3 0f 1e fa          	endbr64 
    3234:	41 57                	push   %r15
    3236:	48 89 f7             	mov    %rsi,%rdi
    3239:	48 8d 35 7b 11 00 00 	lea    0x117b(%rip),%rsi        # 43bb <_fini+0x9db>
    3240:	41 56                	push   %r14
    3242:	41 55                	push   %r13
    3244:	41 54                	push   %r12
    3246:	55                   	push   %rbp
    3247:	53                   	push   %rbx
    3248:	48 81 ec 18 05 00 00 	sub    $0x518,%rsp
    324f:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    3256:	00 00 
    3258:	48 89 84 24 08 05 00 	mov    %rax,0x508(%rsp)
    325f:	00 
    3260:	31 c0                	xor    %eax,%eax
    3262:	48 8d 44 24 34       	lea    0x34(%rsp),%rax
    3267:	48 8d 4c 24 28       	lea    0x28(%rsp),%rcx
    326c:	48 8d 54 24 38       	lea    0x38(%rsp),%rdx
    3271:	50                   	push   %rax
    3272:	48 8d 44 24 58       	lea    0x58(%rsp),%rax
    3277:	50                   	push   %rax
    3278:	48 8d 44 24 58       	lea    0x58(%rsp),%rax
    327d:	50                   	push   %rax
    327e:	48 8d 44 24 58       	lea    0x58(%rsp),%rax
    3283:	50                   	push   %rax
    3284:	31 c0                	xor    %eax,%eax
    3286:	4c 8d 4c 24 50       	lea    0x50(%rsp),%r9
    328b:	4c 8d 44 24 4c       	lea    0x4c(%rsp),%r8
    3290:	e8 eb f1 ff ff       	call   2480 <_PyArg_ParseTuple_SizeT@plt>
    3295:	48 83 c4 20          	add    $0x20,%rsp
    3299:	85 c0                	test   %eax,%eax
    329b:	0f 84 07 01 00 00    	je     33a8 <fillTMADescriptor+0x178>
    32a1:	31 f6                	xor    %esi,%esi
    32a3:	48 8d 3d 36 30 00 00 	lea    0x3036(%rip),%rdi        # 62e0 <PyCUtensorMapType>
    32aa:	e8 c1 f1 ff ff       	call   2470 <PyObject_CallObject@plt>
    32af:	49 89 c5             	mov    %rax,%r13
    32b2:	48 85 c0             	test   %rax,%rax
    32b5:	0f 84 ed 00 00 00    	je     33a8 <fillTMADescriptor+0x178>
    32bb:	48 8b 7c 24 40       	mov    0x40(%rsp),%rdi
    32c0:	48 8d 35 fd 10 00 00 	lea    0x10fd(%rip),%rsi        # 43c4 <_fini+0x9e4>
    32c7:	e8 a4 f2 ff ff       	call   2570 <PySequence_Fast@plt>
    32cc:	49 89 c7             	mov    %rax,%r15
    32cf:	48 85 c0             	test   %rax,%rax
    32d2:	0f 84 c5 00 00 00    	je     339d <fillTMADescriptor+0x16d>
    32d8:	48 8b 40 08          	mov    0x8(%rax),%rax
    32dc:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    32e3:	a9 00 00 00 02       	test   $0x2000000,%eax
    32e8:	75 0b                	jne    32f5 <fillTMADescriptor+0xc5>
    32ea:	a9 00 00 00 04       	test   $0x4000000,%eax
    32ef:	0f 84 ad 02 00 00    	je     35a2 <fillTMADescriptor+0x372>
    32f5:	45 8b 77 10          	mov    0x10(%r15),%r14d
    32f9:	45 85 f6             	test   %r14d,%r14d
    32fc:	0f 8e de 00 00 00    	jle    33e0 <fillTMADescriptor+0x1b0>
    3302:	49 63 d6             	movslq %r14d,%rdx
    3305:	45 8d 66 ff          	lea    -0x1(%r14),%r12d
    3309:	31 db                	xor    %ebx,%ebx
    330b:	48 8d 6c 94 60       	lea    0x60(%rsp,%rdx,4),%rbp
    3310:	49 c1 e4 03          	shl    $0x3,%r12
    3314:	eb 43                	jmp    3359 <fillTMADescriptor+0x129>
    3316:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    331d:	00 00 00 
    3320:	49 8b 47 18          	mov    0x18(%r15),%rax
    3324:	48 8b 3c 18          	mov    (%rax,%rbx,1),%rdi
    3328:	48 8b 47 08          	mov    0x8(%rdi),%rax
    332c:	f6 80 ab 00 00 00 01 	testb  $0x1,0xab(%rax)
    3333:	74 48                	je     337d <fillTMADescriptor+0x14d>
    3335:	e8 b6 f1 ff ff       	call   24f0 <PyLong_AsLongLong@plt>
    333a:	48 83 ed 04          	sub    $0x4,%rbp
    333e:	89 45 00             	mov    %eax,0x0(%rbp)
    3341:	49 39 dc             	cmp    %rbx,%r12
    3344:	0f 84 96 00 00 00    	je     33e0 <fillTMADescriptor+0x1b0>
    334a:	49 8b 47 08          	mov    0x8(%r15),%rax
    334e:	48 83 c3 08          	add    $0x8,%rbx
    3352:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    3359:	a9 00 00 00 02       	test   $0x2000000,%eax
    335e:	75 c0                	jne    3320 <fillTMADescriptor+0xf0>
    3360:	a9 00 00 00 04       	test   $0x4000000,%eax
    3365:	0f 84 3d 05 00 00    	je     38a8 <fillTMADescriptor+0x678>
    336b:	49 8b 7c 1f 18       	mov    0x18(%r15,%rbx,1),%rdi
    3370:	48 8b 47 08          	mov    0x8(%rdi),%rax
    3374:	f6 80 ab 00 00 00 01 	testb  $0x1,0xab(%rax)
    337b:	75 b8                	jne    3335 <fillTMADescriptor+0x105>
    337d:	48 8b 05 54 2c 00 00 	mov    0x2c54(%rip),%rax        # 5fd8 <PyExc_TypeError@Base>
    3384:	48 8d 35 91 10 00 00 	lea    0x1091(%rip),%rsi        # 441c <_fini+0xa3c>
    338b:	48 8b 38             	mov    (%rax),%rdi
    338e:	e8 2d f0 ff ff       	call   23c0 <PyErr_SetString@plt>
    3393:	49 83 2f 01          	subq   $0x1,(%r15)
    3397:	0f 84 53 01 00 00    	je     34f0 <fillTMADescriptor+0x2c0>
    339d:	49 83 6d 00 01       	subq   $0x1,0x0(%r13)
    33a2:	0f 84 10 01 00 00    	je     34b8 <fillTMADescriptor+0x288>
    33a8:	45 31 ed             	xor    %r13d,%r13d
    33ab:	48 8b 84 24 08 05 00 	mov    0x508(%rsp),%rax
    33b2:	00 
    33b3:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    33ba:	00 00 
    33bc:	0f 85 05 05 00 00    	jne    38c7 <fillTMADescriptor+0x697>
    33c2:	48 81 c4 18 05 00 00 	add    $0x518,%rsp
    33c9:	4c 89 e8             	mov    %r13,%rax
    33cc:	5b                   	pop    %rbx
    33cd:	5d                   	pop    %rbp
    33ce:	41 5c                	pop    %r12
    33d0:	41 5d                	pop    %r13
    33d2:	41 5e                	pop    %r14
    33d4:	41 5f                	pop    %r15
    33d6:	c3                   	ret    
    33d7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    33de:	00 00 
    33e0:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    33e5:	48 8d 35 4a 10 00 00 	lea    0x104a(%rip),%rsi        # 4436 <_fini+0xa56>
    33ec:	e8 7f f1 ff ff       	call   2570 <PySequence_Fast@plt>
    33f1:	49 89 c4             	mov    %rax,%r12
    33f4:	48 85 c0             	test   %rax,%rax
    33f7:	0f 84 e9 00 00 00    	je     34e6 <fillTMADescriptor+0x2b6>
    33fd:	49 63 c6             	movslq %r14d,%rax
    3400:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    3405:	49 8b 44 24 08       	mov    0x8(%r12),%rax
    340a:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    3411:	a9 00 00 00 02       	test   $0x2000000,%eax
    3416:	0f 84 ac 00 00 00    	je     34c8 <fillTMADescriptor+0x298>
    341c:	49 8b 54 24 10       	mov    0x10(%r12),%rdx
    3421:	49 63 ce             	movslq %r14d,%rcx
    3424:	48 39 d1             	cmp    %rdx,%rcx
    3427:	0f 85 d0 00 00 00    	jne    34fd <fillTMADescriptor+0x2cd>
    342d:	45 85 f6             	test   %r14d,%r14d
    3430:	0f 8e 3a 01 00 00    	jle    3570 <fillTMADescriptor+0x340>
    3436:	41 8d 56 ff          	lea    -0x1(%r14),%edx
    343a:	48 8d ac cc a0 00 00 	lea    0xa0(%rsp,%rcx,8),%rbp
    3441:	00 
    3442:	31 db                	xor    %ebx,%ebx
    3444:	48 8d 0c d5 00 00 00 	lea    0x0(,%rdx,8),%rcx
    344b:	00 
    344c:	48 89 0c 24          	mov    %rcx,(%rsp)
    3450:	eb 47                	jmp    3499 <fillTMADescriptor+0x269>
    3452:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3458:	49 8b 44 24 18       	mov    0x18(%r12),%rax
    345d:	48 8b 3c 18          	mov    (%rax,%rbx,1),%rdi
    3461:	48 8b 47 08          	mov    0x8(%rdi),%rax
    3465:	f6 80 ab 00 00 00 01 	testb  $0x1,0xab(%rax)
    346c:	0f 84 ee 01 00 00    	je     3660 <fillTMADescriptor+0x430>
    3472:	e8 e9 ef ff ff       	call   2460 <PyLong_AsLong@plt>
    3477:	48 83 ed 08          	sub    $0x8,%rbp
    347b:	48 89 45 00          	mov    %rax,0x0(%rbp)
    347f:	48 39 1c 24          	cmp    %rbx,(%rsp)
    3483:	0f 84 e7 00 00 00    	je     3570 <fillTMADescriptor+0x340>
    3489:	49 8b 44 24 08       	mov    0x8(%r12),%rax
    348e:	48 83 c3 08          	add    $0x8,%rbx
    3492:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    3499:	a9 00 00 00 02       	test   $0x2000000,%eax
    349e:	75 b8                	jne    3458 <fillTMADescriptor+0x228>
    34a0:	a9 00 00 00 04       	test   $0x4000000,%eax
    34a5:	0f 84 21 04 00 00    	je     38cc <fillTMADescriptor+0x69c>
    34ab:	49 8b 7c 1c 18       	mov    0x18(%r12,%rbx,1),%rdi
    34b0:	eb af                	jmp    3461 <fillTMADescriptor+0x231>
    34b2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    34b8:	4c 89 ef             	mov    %r13,%rdi
    34bb:	e8 c0 ee ff ff       	call   2380 <_Py_Dealloc@plt>
    34c0:	e9 e3 fe ff ff       	jmp    33a8 <fillTMADescriptor+0x178>
    34c5:	0f 1f 00             	nopl   (%rax)
    34c8:	a9 00 00 00 04       	test   $0x4000000,%eax
    34cd:	0f 84 cf 00 00 00    	je     35a2 <fillTMADescriptor+0x372>
    34d3:	49 8b 54 24 10       	mov    0x10(%r12),%rdx
    34d8:	49 63 ce             	movslq %r14d,%rcx
    34db:	48 39 d1             	cmp    %rdx,%rcx
    34de:	0f 84 49 ff ff ff    	je     342d <fillTMADescriptor+0x1fd>
    34e4:	eb 17                	jmp    34fd <fillTMADescriptor+0x2cd>
    34e6:	49 83 2f 01          	subq   $0x1,(%r15)
    34ea:	0f 85 ad fe ff ff    	jne    339d <fillTMADescriptor+0x16d>
    34f0:	4c 89 ff             	mov    %r15,%rdi
    34f3:	e8 88 ee ff ff       	call   2380 <_Py_Dealloc@plt>
    34f8:	e9 a0 fe ff ff       	jmp    339d <fillTMADescriptor+0x16d>
    34fd:	48 8b 05 c4 2a 00 00 	mov    0x2ac4(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    3504:	48 8d 35 44 0f 00 00 	lea    0xf44(%rip),%rsi        # 444f <_fini+0xa6f>
    350b:	31 ed                	xor    %ebp,%ebp
    350d:	48 8b 38             	mov    (%rax),%rdi
    3510:	e8 ab ee ff ff       	call   23c0 <PyErr_SetString@plt>
    3515:	49 83 2f 01          	subq   $0x1,(%r15)
    3519:	75 08                	jne    3523 <fillTMADescriptor+0x2f3>
    351b:	4c 89 ff             	mov    %r15,%rdi
    351e:	e8 5d ee ff ff       	call   2380 <_Py_Dealloc@plt>
    3523:	49 83 2c 24 01       	subq   $0x1,(%r12)
    3528:	74 36                	je     3560 <fillTMADescriptor+0x330>
    352a:	48 85 ed             	test   %rbp,%rbp
    352d:	0f 84 6a fe ff ff    	je     339d <fillTMADescriptor+0x16d>
    3533:	48 83 6d 00 01       	subq   $0x1,0x0(%rbp)
    3538:	0f 85 5f fe ff ff    	jne    339d <fillTMADescriptor+0x16d>
    353e:	48 89 ef             	mov    %rbp,%rdi
    3541:	e8 3a ee ff ff       	call   2380 <_Py_Dealloc@plt>
    3546:	49 83 6d 00 01       	subq   $0x1,0x0(%r13)
    354b:	0f 85 57 fe ff ff    	jne    33a8 <fillTMADescriptor+0x178>
    3551:	e9 62 ff ff ff       	jmp    34b8 <fillTMADescriptor+0x288>
    3556:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    355d:	00 00 00 
    3560:	4c 89 e7             	mov    %r12,%rdi
    3563:	e8 18 ee ff ff       	call   2380 <_Py_Dealloc@plt>
    3568:	eb c0                	jmp    352a <fillTMADescriptor+0x2fa>
    356a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3570:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    3575:	48 8d 35 0f 0f 00 00 	lea    0xf0f(%rip),%rsi        # 448b <_fini+0xaab>
    357c:	e8 ef ef ff ff       	call   2570 <PySequence_Fast@plt>
    3581:	48 89 c5             	mov    %rax,%rbp
    3584:	48 85 c0             	test   %rax,%rax
    3587:	74 8c                	je     3515 <fillTMADescriptor+0x2e5>
    3589:	48 8b 40 08          	mov    0x8(%rax),%rax
    358d:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    3594:	a9 00 00 00 02       	test   $0x2000000,%eax
    3599:	75 15                	jne    35b0 <fillTMADescriptor+0x380>
    359b:	a9 00 00 00 04       	test   $0x4000000,%eax
    35a0:	75 0e                	jne    35b0 <fillTMADescriptor+0x380>
    35a2:	e8 39 f1 ff ff       	call   26e0 <PyTuple_GET_SIZE.part.0>
    35a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    35ae:	00 00 
    35b0:	48 8b 55 10          	mov    0x10(%rbp),%rdx
    35b4:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    35b9:	48 39 d1             	cmp    %rdx,%rcx
    35bc:	0f 85 bb 00 00 00    	jne    367d <fillTMADescriptor+0x44d>
    35c2:	41 83 fe 01          	cmp    $0x1,%r14d
    35c6:	0f 8e 3d 02 00 00    	jle    3809 <fillTMADescriptor+0x5d9>
    35cc:	41 8d 56 fe          	lea    -0x2(%r14),%edx
    35d0:	48 8d 8c cc d0 00 00 	lea    0xd0(%rsp,%rcx,8),%rcx
    35d7:	00 
    35d8:	89 54 24 1c          	mov    %edx,0x1c(%rsp)
    35dc:	48 c1 e2 03          	shl    $0x3,%rdx
    35e0:	48 89 54 24 10       	mov    %rdx,0x10(%rsp)
    35e5:	31 d2                	xor    %edx,%edx
    35e7:	eb 5e                	jmp    3647 <fillTMADescriptor+0x417>
    35e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    35f0:	48 8b 45 18          	mov    0x18(%rbp),%rax
    35f4:	48 8b 3c 10          	mov    (%rax,%rdx,1),%rdi
    35f8:	48 8b 47 08          	mov    0x8(%rdi),%rax
    35fc:	f6 80 ab 00 00 00 01 	testb  $0x1,0xab(%rax)
    3603:	0f 84 8f 00 00 00    	je     3698 <fillTMADescriptor+0x468>
    3609:	48 63 5c 24 2c       	movslq 0x2c(%rsp),%rbx
    360e:	48 89 4c 24 08       	mov    %rcx,0x8(%rsp)
    3613:	48 89 14 24          	mov    %rdx,(%rsp)
    3617:	e8 d4 ee ff ff       	call   24f0 <PyLong_AsLongLong@plt>
    361c:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    3621:	48 8b 14 24          	mov    (%rsp),%rdx
    3625:	48 0f af d8          	imul   %rax,%rbx
    3629:	48 83 e9 08          	sub    $0x8,%rcx
    362d:	48 89 59 f8          	mov    %rbx,-0x8(%rcx)
    3631:	48 3b 54 24 10       	cmp    0x10(%rsp),%rdx
    3636:	74 7b                	je     36b3 <fillTMADescriptor+0x483>
    3638:	48 8b 45 08          	mov    0x8(%rbp),%rax
    363c:	48 83 c2 08          	add    $0x8,%rdx
    3640:	48 8b 80 a8 00 00 00 	mov    0xa8(%rax),%rax
    3647:	a9 00 00 00 02       	test   $0x2000000,%eax
    364c:	75 a2                	jne    35f0 <fillTMADescriptor+0x3c0>
    364e:	a9 00 00 00 04       	test   $0x4000000,%eax
    3653:	0f 84 f3 02 00 00    	je     394c <fillTMADescriptor+0x71c>
    3659:	48 8b 7c 15 18       	mov    0x18(%rbp,%rdx,1),%rdi
    365e:	eb 98                	jmp    35f8 <fillTMADescriptor+0x3c8>
    3660:	48 8b 05 71 29 00 00 	mov    0x2971(%rip),%rax        # 5fd8 <PyExc_TypeError@Base>
    3667:	48 8d 35 08 0e 00 00 	lea    0xe08(%rip),%rsi        # 4476 <_fini+0xa96>
    366e:	31 ed                	xor    %ebp,%ebp
    3670:	48 8b 38             	mov    (%rax),%rdi
    3673:	e8 48 ed ff ff       	call   23c0 <PyErr_SetString@plt>
    3678:	e9 98 fe ff ff       	jmp    3515 <fillTMADescriptor+0x2e5>
    367d:	48 8b 05 44 29 00 00 	mov    0x2944(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    3684:	48 8d 35 c4 0d 00 00 	lea    0xdc4(%rip),%rsi        # 444f <_fini+0xa6f>
    368b:	48 8b 38             	mov    (%rax),%rdi
    368e:	e8 2d ed ff ff       	call   23c0 <PyErr_SetString@plt>
    3693:	e9 7d fe ff ff       	jmp    3515 <fillTMADescriptor+0x2e5>
    3698:	48 8b 05 39 29 00 00 	mov    0x2939(%rip),%rax        # 5fd8 <PyExc_TypeError@Base>
    369f:	48 8d 35 d0 0d 00 00 	lea    0xdd0(%rip),%rsi        # 4476 <_fini+0xa96>
    36a6:	48 8b 38             	mov    (%rax),%rdi
    36a9:	e8 12 ed ff ff       	call   23c0 <PyErr_SetString@plt>
    36ae:	e9 62 fe ff ff       	jmp    3515 <fillTMADescriptor+0x2e5>
    36b3:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    36b7:	48 98                	cltq   
    36b9:	48 8b 8c c4 a0 00 00 	mov    0xa0(%rsp,%rax,8),%rcx
    36c0:	00 
    36c1:	48 63 54 24 1c       	movslq 0x1c(%rsp),%rdx
    36c6:	48 8b 94 d4 d0 00 00 	mov    0xd0(%rsp,%rdx,8),%rdx
    36cd:	00 
    36ce:	48 0f af d1          	imul   %rcx,%rdx
    36d2:	48 89 94 c4 d0 00 00 	mov    %rdx,0xd0(%rsp,%rax,8)
    36d9:	00 
    36da:	49 83 2f 01          	subq   $0x1,(%r15)
    36de:	0f 84 5f 01 00 00    	je     3843 <fillTMADescriptor+0x613>
    36e4:	49 83 2c 24 01       	subq   $0x1,(%r12)
    36e9:	0f 84 47 01 00 00    	je     3836 <fillTMADescriptor+0x606>
    36ef:	48 83 6d 00 01       	subq   $0x1,0x0(%rbp)
    36f4:	0f 84 2f 01 00 00    	je     3829 <fillTMADescriptor+0x5f9>
    36fa:	66 0f 6f 05 ae 0e 00 	movdqa 0xeae(%rip),%xmm0        # 45b0 <__PRETTY_FUNCTION__.4+0x30>
    3701:	00 
    3702:	45 31 e4             	xor    %r12d,%r12d
    3705:	48 8b 1d 7c 2d 00 00 	mov    0x2d7c(%rip),%rbx        # 6488 <cuTensorMapEncodeTiled.3>
    370c:	c7 84 24 90 00 00 00 	movl   $0x1,0x90(%rsp)
    3713:	01 00 00 00 
    3717:	83 7c 24 34 01       	cmpl   $0x1,0x34(%rsp)
    371c:	41 0f 94 c4          	sete   %r12b
    3720:	0f 29 84 24 80 00 00 	movaps %xmm0,0x80(%rsp)
    3727:	00 
    3728:	48 85 db             	test   %rbx,%rbx
    372b:	0f 84 1f 01 00 00    	je     3850 <fillTMADescriptor+0x620>
    3731:	41 54                	push   %r12
    3733:	49 8d bd 80 00 00 00 	lea    0x80(%r13),%rdi
    373a:	44 89 f2             	mov    %r14d,%edx
    373d:	6a 02                	push   $0x2
    373f:	8b 44 24 38          	mov    0x38(%rsp),%eax
    3743:	50                   	push   %rax
    3744:	6a 00                	push   $0x0
    3746:	48 8d 84 24 a0 00 00 	lea    0xa0(%rsp),%rax
    374d:	00 
    374e:	50                   	push   %rax
    374f:	48 8d 84 24 88 00 00 	lea    0x88(%rsp),%rax
    3756:	00 
    3757:	50                   	push   %rax
    3758:	48 8b 4c 24 68       	mov    0x68(%rsp),%rcx
    375d:	8b 74 24 60          	mov    0x60(%rsp),%esi
    3761:	4c 8d 8c 24 00 01 00 	lea    0x100(%rsp),%r9
    3768:	00 
    3769:	4c 8d 84 24 d0 00 00 	lea    0xd0(%rsp),%r8
    3770:	00 
    3771:	ff d3                	call   *%rbx
    3773:	48 83 c4 30          	add    $0x30,%rsp
    3777:	89 c7                	mov    %eax,%edi
    3779:	85 c0                	test   %eax,%eax
    377b:	0f 84 2a fc ff ff    	je     33ab <fillTMADescriptor+0x17b>
    3781:	48 8d 74 24 58       	lea    0x58(%rsp),%rsi
    3786:	4c 8d a4 24 00 01 00 	lea    0x100(%rsp),%r12
    378d:	00 
    378e:	e8 0d ed ff ff       	call   24a0 <cuGetErrorString@plt>
    3793:	66 0f 6f 05 05 0e 00 	movdqa 0xe05(%rip),%xmm0        # 45a0 <__PRETTY_FUNCTION__.4+0x20>
    379a:	00 
    379b:	31 c0                	xor    %eax,%eax
    379d:	48 8b 74 24 58       	mov    0x58(%rsp),%rsi
    37a2:	48 8d bc 24 10 01 00 	lea    0x110(%rsp),%rdi
    37a9:	00 
    37aa:	b9 7e 00 00 00       	mov    $0x7e,%ecx
    37af:	ba eb 03 00 00       	mov    $0x3eb,%edx
    37b4:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    37b7:	48 8d bc 24 15 01 00 	lea    0x115(%rsp),%rdi
    37be:	00 
    37bf:	b8 20 00 00 00       	mov    $0x20,%eax
    37c4:	0f 29 84 24 00 01 00 	movaps %xmm0,0x100(%rsp)
    37cb:	00 
    37cc:	c7 84 24 10 01 00 00 	movl   $0x3a5d4144,0x110(%rsp)
    37d3:	44 41 5d 3a 
    37d7:	66 89 84 24 14 01 00 	mov    %ax,0x114(%rsp)
    37de:	00 
    37df:	e8 cc ec ff ff       	call   24b0 <__strcpy_chk@plt>
    37e4:	e8 f7 eb ff ff       	call   23e0 <PyGILState_Ensure@plt>
    37e9:	4c 89 e6             	mov    %r12,%rsi
    37ec:	89 c5                	mov    %eax,%ebp
    37ee:	48 8b 05 d3 27 00 00 	mov    0x27d3(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    37f5:	48 8b 38             	mov    (%rax),%rdi
    37f8:	e8 c3 eb ff ff       	call   23c0 <PyErr_SetString@plt>
    37fd:	89 ef                	mov    %ebp,%edi
    37ff:	e8 5c eb ff ff       	call   2360 <PyGILState_Release@plt>
    3804:	e9 94 fb ff ff       	jmp    339d <fillTMADescriptor+0x16d>
    3809:	41 8d 46 ff          	lea    -0x1(%r14),%eax
    380d:	48 98                	cltq   
    380f:	48 8b 8c c4 a0 00 00 	mov    0xa0(%rsp,%rax,8),%rcx
    3816:	00 
    3817:	0f 85 22 01 00 00    	jne    393f <fillTMADescriptor+0x70f>
    381d:	48 63 54 24 2c       	movslq 0x2c(%rsp),%rdx
    3822:	31 c0                	xor    %eax,%eax
    3824:	e9 a5 fe ff ff       	jmp    36ce <fillTMADescriptor+0x49e>
    3829:	48 89 ef             	mov    %rbp,%rdi
    382c:	e8 4f eb ff ff       	call   2380 <_Py_Dealloc@plt>
    3831:	e9 c4 fe ff ff       	jmp    36fa <fillTMADescriptor+0x4ca>
    3836:	4c 89 e7             	mov    %r12,%rdi
    3839:	e8 42 eb ff ff       	call   2380 <_Py_Dealloc@plt>
    383e:	e9 ac fe ff ff       	jmp    36ef <fillTMADescriptor+0x4bf>
    3843:	4c 89 ff             	mov    %r15,%rdi
    3846:	e8 35 eb ff ff       	call   2380 <_Py_Dealloc@plt>
    384b:	e9 94 fe ff ff       	jmp    36e4 <fillTMADescriptor+0x4b4>
    3850:	be 01 00 00 00       	mov    $0x1,%esi
    3855:	48 8d 3d 35 0b 00 00 	lea    0xb35(%rip),%rdi        # 4391 <_fini+0x9b1>
    385c:	e8 bf eb ff ff       	call   2420 <dlopen@plt>
    3861:	48 89 c5             	mov    %rax,%rbp
    3864:	48 85 c0             	test   %rax,%rax
    3867:	0f 84 ac 00 00 00    	je     3919 <fillTMADescriptor+0x6e9>
    386d:	e8 ae ea ff ff       	call   2320 <dlerror@plt>
    3872:	48 8d 35 48 0c 00 00 	lea    0xc48(%rip),%rsi        # 44c1 <_fini+0xae1>
    3879:	48 89 ef             	mov    %rbp,%rdi
    387c:	e8 5f ec ff ff       	call   24e0 <dlsym@plt>
    3881:	48 89 c3             	mov    %rax,%rbx
    3884:	e8 97 ea ff ff       	call   2320 <dlerror@plt>
    3889:	48 85 c0             	test   %rax,%rax
    388c:	75 5d                	jne    38eb <fillTMADescriptor+0x6bb>
    388e:	48 89 1d f3 2b 00 00 	mov    %rbx,0x2bf3(%rip)        # 6488 <cuTensorMapEncodeTiled.3>
    3895:	48 85 db             	test   %rbx,%rbx
    3898:	0f 85 93 fe ff ff    	jne    3731 <fillTMADescriptor+0x501>
    389e:	e9 fa fa ff ff       	jmp    339d <fillTMADescriptor+0x16d>
    38a3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    38a8:	48 8d 0d d1 0c 00 00 	lea    0xcd1(%rip),%rcx        # 4580 <__PRETTY_FUNCTION__.4>
    38af:	ba 6d 01 00 00       	mov    $0x16d,%edx
    38b4:	48 8d 35 26 0b 00 00 	lea    0xb26(%rip),%rsi        # 43e1 <_fini+0xa01>
    38bb:	48 8d 3d 3d 0b 00 00 	lea    0xb3d(%rip),%rdi        # 43ff <_fini+0xa1f>
    38c2:	e8 09 eb ff ff       	call   23d0 <__assert_fail@plt>
    38c7:	e8 e4 ea ff ff       	call   23b0 <__stack_chk_fail@plt>
    38cc:	48 8d 0d ad 0c 00 00 	lea    0xcad(%rip),%rcx        # 4580 <__PRETTY_FUNCTION__.4>
    38d3:	ba 7e 01 00 00       	mov    $0x17e,%edx
    38d8:	48 8d 35 02 0b 00 00 	lea    0xb02(%rip),%rsi        # 43e1 <_fini+0xa01>
    38df:	48 8d 3d 77 0b 00 00 	lea    0xb77(%rip),%rdi        # 445d <_fini+0xa7d>
    38e6:	e8 e5 ea ff ff       	call   23d0 <__assert_fail@plt>
    38eb:	48 8b 05 d6 26 00 00 	mov    0x26d6(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    38f2:	48 8d 35 cf 07 00 00 	lea    0x7cf(%rip),%rsi        # 40c8 <_fini+0x6e8>
    38f9:	48 8b 38             	mov    (%rax),%rdi
    38fc:	e8 bf ea ff ff       	call   23c0 <PyErr_SetString@plt>
    3901:	48 89 ef             	mov    %rbp,%rdi
    3904:	e8 47 ec ff ff       	call   2550 <dlclose@plt>
    3909:	48 c7 05 74 2b 00 00 	movq   $0x0,0x2b74(%rip)        # 6488 <cuTensorMapEncodeTiled.3>
    3910:	00 00 00 00 
    3914:	e9 84 fa ff ff       	jmp    339d <fillTMADescriptor+0x16d>
    3919:	48 8b 05 a8 26 00 00 	mov    0x26a8(%rip),%rax        # 5fc8 <PyExc_RuntimeError@Base>
    3920:	48 8d 35 5b 0a 00 00 	lea    0xa5b(%rip),%rsi        # 4382 <_fini+0x9a2>
    3927:	48 8b 38             	mov    (%rax),%rdi
    392a:	e8 91 ea ff ff       	call   23c0 <PyErr_SetString@plt>
    392f:	48 c7 05 4e 2b 00 00 	movq   $0x0,0x2b4e(%rip)        # 6488 <cuTensorMapEncodeTiled.3>
    3936:	00 00 00 00 
    393a:	e9 5e fa ff ff       	jmp    339d <fillTMADescriptor+0x16d>
    393f:	41 8d 56 fe          	lea    -0x2(%r14),%edx
    3943:	89 54 24 1c          	mov    %edx,0x1c(%rsp)
    3947:	e9 75 fd ff ff       	jmp    36c1 <fillTMADescriptor+0x491>
    394c:	48 8d 0d 2d 0c 00 00 	lea    0xc2d(%rip),%rcx        # 4580 <__PRETTY_FUNCTION__.4>
    3953:	ba 8f 01 00 00       	mov    $0x18f,%edx
    3958:	48 8d 35 82 0a 00 00 	lea    0xa82(%rip),%rsi        # 43e1 <_fini+0xa01>
    395f:	48 8d 3d 40 0b 00 00 	lea    0xb40(%rip),%rdi        # 44a6 <_fini+0xac6>
    3966:	e8 65 ea ff ff       	call   23d0 <__assert_fail@plt>
    396b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000003970 <PyInit_cuda_utils>:
    3970:	f3 0f 1e fa          	endbr64 
    3974:	41 55                	push   %r13
    3976:	4c 8d 2d 63 29 00 00 	lea    0x2963(%rip),%r13        # 62e0 <PyCUtensorMapType>
    397d:	41 54                	push   %r12
    397f:	4c 89 ef             	mov    %r13,%rdi
    3982:	45 31 e4             	xor    %r12d,%r12d
    3985:	48 83 ec 08          	sub    $0x8,%rsp
    3989:	e8 72 ea ff ff       	call   2400 <PyType_Ready@plt>
    398e:	85 c0                	test   %eax,%eax
    3990:	78 42                	js     39d4 <PyInit_cuda_utils+0x64>
    3992:	be f5 03 00 00       	mov    $0x3f5,%esi
    3997:	48 8d 3d 02 28 00 00 	lea    0x2802(%rip),%rdi        # 61a0 <ModuleDef>
    399e:	e8 ad ea ff ff       	call   2450 <PyModule_Create2@plt>
    39a3:	49 89 c4             	mov    %rax,%r12
    39a6:	48 85 c0             	test   %rax,%rax
    39a9:	74 29                	je     39d4 <PyInit_cuda_utils+0x64>
    39ab:	48 8d 35 6e 28 00 00 	lea    0x286e(%rip),%rsi        # 6220 <ModuleMethods>
    39b2:	48 89 c7             	mov    %rax,%rdi
    39b5:	e8 d6 e9 ff ff       	call   2390 <PyModule_AddFunctions@plt>
    39ba:	4c 89 ea             	mov    %r13,%rdx
    39bd:	4c 89 e7             	mov    %r12,%rdi
    39c0:	48 8d 35 11 0b 00 00 	lea    0xb11(%rip),%rsi        # 44d8 <_fini+0xaf8>
    39c7:	48 83 05 11 29 00 00 	addq   $0x1,0x2911(%rip)        # 62e0 <PyCUtensorMapType>
    39ce:	01 
    39cf:	e8 fc ea ff ff       	call   24d0 <PyModule_AddObject@plt>
    39d4:	48 83 c4 08          	add    $0x8,%rsp
    39d8:	4c 89 e0             	mov    %r12,%rax
    39db:	41 5c                	pop    %r12
    39dd:	41 5d                	pop    %r13
    39df:	c3                   	ret    

Disassembly of section .fini:

00000000000039e0 <_fini>:
    39e0:	f3 0f 1e fa          	endbr64 
    39e4:	48 83 ec 08          	sub    $0x8,%rsp
    39e8:	48 83 c4 08          	add    $0x8,%rsp
    39ec:	c3                   	ret    
