
n
volta_sgemm_32x32_sliced1x4_nnV*28ž
@8HPXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  A
C
volta_sgemm_128x32_nn9*28’×@0HXbCudnnRNNhu  HB
L
volta_sgemm_32x32_sliced1x4_nnV*28@8HhbCudnnRNNh
u  A
n
volta_sgemm_32x32_sliced1x4_nnV*28’ē@’7HHXb(gradients/CudnnRNN_grad/CudnnRNNBackproph
u  A

Žvoid GRU_elementWise_fp<float, float, float, (cudnnRNNBiasMode_t)2>(int, int, int, int, float const*, float const*, float const*, float const*, cudnn::reduced_divisor, float*, float const*, float*, float*, bool, bool, int) *28’×@H(bCudnnRNNhu  ČB
Å
yvoid GRU_elementWise_bp1<float, float, float>(int, int, float*, float*, float*, float*, float*, float*, float*, int, int)*28°@H Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  ČB
Ņ
void GRU_elementWise_bp2<float, float, float, 3, (cudnnRNNBiasMode_t)2>(int, int, float*, float*, cudnn::reduced_divisor, float*)'`*  28’ß@čH’÷Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  HB
é
Øvoid cub::DeviceRadixSortSingleTileKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, false, int, int, int>(int const*, int*, int const*, int*, int, int, int)Ą*28č@čHčbAdam/Adam/update/Uniquehu  ČA
n
volta_sgemm_32x32_sliced1x4_ntV*28ø@HHpXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  śA
š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)2*28°@°H°b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
ŗ
void transpose_readWrite_alignment_kernel<float, float, 1, false, 6, 5, 3>(cublasTransposeParams<float>, float const*, float*, float const*)0A*28°@ H@bCudnnRNNhu  zB
n
volta_sgemm_32x32_sliced1x4_ntV*28Ø@PHXXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  śA
n
volta_sgemm_64x32_sliced1x4_ntRĄ*28Ø@@HhXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  ČA
P
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28 @ H bAdam/Cast_1hu  ČB
ė
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*28@H bgradients/split_1_grad/concathu  ČB
é
«void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*28@Hbgradients/split_grad/concathu  ČB
ė
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::MaxReducer<float, 0>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)@*28p@pHpb:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
j
volta_sgemm_32x32_sliced1x4_ntV*28’o@’oH’ob'gradient_tape/sequential/dense/MatMul_1hu  śA
å
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorReshapingOp<Eigen::IndexList<int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> >, Eigen::TensorReductionOp<Eigen::internal::SumReducer<float>, Eigen::IndexList<Eigen::type2index<1l>> const, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)1*28`@`H`b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
­
void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*28`@(H8bsplithu  ČB
S
volta_sgemm_128x32_nn9*28`@`H`Xbsequential/dense/MatMulhu  HB
Æ
void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*28X@(H0bsplit_1hu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28X@ H8b$gradients/transpose_2_grad/transposehu  ČB
Æ
void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*28P@(H(bsplit_2hu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28P@ H0btranspose_1hu  ČB
a
volta_sgemm_128x32_tn9*28P@PHPXb%gradient_tape/sequential/dense/MatMulhu  HB
Ģ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8H@ H(btranspose_7hu  ČB
ć
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8H@ H(b"gradients/transpose_grad/transposehu  ČB
“
ųvoid cub::DeviceScanKernel<cub::AgentScanPolicy<128, 15, int, (cub::BlockLoadAlgorithm)2, (cub::CacheLoadModifier)0, (cub::BlockStoreAlgorithm)2, (cub::BlockScanAlgorithm)2, cub::MemBoundScaling<128, 15, int> >, cub::TransformInputIterator<int, tensorflow::unique_op_gpu::SegmentIndicatorFunctor<int, int>, cub::CountingInputIterator<int, long>, long>, int*, cub::ScanTileState<int, true>, cub::Sum, cub::NullType, int>(cub::TransformInputIterator<int, tensorflow::unique_op_gpu::SegmentIndicatorFunctor<int, int>, cub::CountingInputIterator<int, long>, long>, int*, cub::ScanTileState<int, true>, int, cub::Sum, cub::NullType, int)(<*28H@HHHbAdam/Adam/update/Uniquehu  B

ļvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8@@ H bgradients/AddNhu  ČB
č
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *2(8@@ H b-gradients/strided_slice_grad/StridedSliceGradhu  ČB
å
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2(8@@ H b$gradients/transpose_7_grad/transposehu  ČB
Ķ
Øvoid cub::DeviceRadixSortSingleTileKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, false, int, int, int>(int const*, int*, int const*, int*, int, int, int)Ą*28@@@H@hu  ČA
é
void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *28@@H(Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  ČB
é
void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *208@@ H Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  ČB
§
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *28@@@H@b2gradient_tape/sequential/dense/BiasAdd/BiasAddGradhu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H b$gradients/transpose_5_grad/transposehu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H b$gradients/transpose_6_grad/transposehu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H btranspose_4hu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H btranspose_5hu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H btranspose_6hu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28@@ H b$gradients/transpose_3_grad/transposehu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28’?@’H b$gradients/transpose_4_grad/transposehu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*28’?@’H btranspose_2hu  ČB
Å
évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)/*288@8H8b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
Ä
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*288@H b-gradients/strided_slice_grad/StridedSliceGradhu  ČB
ń
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*288@H b$gradients/transpose_1_grad/transposehu  ČB
’	
£	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)**280@0H0b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
Ė
void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool) *280@0H0bsequential/dense/Softmaxhu  ČB
Ž
void tensorflow::UnsortedSegmentCustomKernel<float, int, tensorflow::functor::AtomicSumOpGpu<float> >(long, long, long, int const*, float const*, float*)*2P80@0H0b#Adam/Adam/update/UnsortedSegmentSumhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *280@0H0b$Adam/Adam/update_4/ResourceApplyAdamhu  ČB
ī
void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*280@0H0b7sequential/dropout/dropout/random_uniform/RandomUniformhu  ČB
Ų
Øvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) !*280@Hbtranspose_3hu  ČB
­
Ńvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorBroadcastingOp<Eigen::array<long, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorBroadcastingOp<Eigen::IndexList<Eigen::type2index<1l>, int> const, Eigen::TensorForcedEvalOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)**28’/@’/H’/b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  HB
_
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(bsequential/bidirectional/addhu  ČB
W
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(bAdam/Adam/update/addhu  ČB
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*28(@(H(b'sequential/dropout/dropout/GreaterEqualhu  ČB
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(b sequential/dropout/dropout/Mul_1hu  ČB
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28(@(H(bAdam/Powhu  ČB
·
ńvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28(@(H(b$Adam/Adam/update/AssignSubVariableOphu  ČB
Å
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8(@(H(b"gradients/ReverseV2_grad/ReverseV2hu  ČB
³
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *28(@(H(bArgMaxhu  ČB
µ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorTupleReducerOp<Eigen::internal::ArgMaxTupleReducer<Eigen::Tuple<long, float> >, Eigen::array<long, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *28(@(H(bArgMax_1hu  ČB
Ź
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8(@(H(b	transposehu  ČB

Åvoid tensorflow::GatherOpKernel<tensorflow::AlignedVector<float, 4>, int, true>(tensorflow::AlignedVector<float, 4> const*, int const*, tensorflow::AlignedVector<float, 4>*, long, long, long, long)*28(@(H(b%sequential/embedding/embedding_lookuphu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_3/ResourceApplyAdamhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_7/ResourceApplyAdamhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_8/ResourceApplyAdamhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_2/ResourceApplyAdamhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_5/ResourceApplyAdamhu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28(@(H(b$Adam/Adam/update_1/ResourceApplyAdamhu  ČB
’
Åvoid tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*2 8(@(H(bsequential/dense/Softmaxhu  ČB
Ż
void tensorflow::scatter_op_gpu::ScatterOpCustomKernel<float, int, (tensorflow::scatter_op::UpdateOp)1>(float*, float const*, int const*, int, int, int)*28(@(H(b#Adam/Adam/update/ResourceScatterAddhu  ČB

tvoid tensorflow::unique_op_gpu::LookupAndScatterUniqueIdsKernel<int>(long, int const*, int const*, int const*, int*)*28(@(H(hu  ČB
ū
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *28’'@’'H’'b$Adam/Adam/update_6/ResourceApplyAdamhu  ČB
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*28 @ H bAdam/addhu  ČB
[
!Cast_GPU_DT_FLOAT_DT_INT32_kernel*28 @ H bsequential/embedding/Casthu  ČB
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28 @ H bCasthu  ČB
Q
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*28 @ H bsequential/Casthu  ČB
Y
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/truedivhu  ČB
G
!Equal_GPU_DT_INT64_DT_BOOL_kernel*28 @ H bEqualhu  ČB
d
!Equal_GPU_DT_INT64_DT_BOOL_kernel*28 @ H b"batch_accuracy/write_summary/Equalhu  ČB

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bLgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mulhu  ČB
U
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/mulhu  ČB
I
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/mulhu  ČB
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b.gradient_tape/sequential/dropout/dropout/Mul_1hu  ČB
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bsequential/dropout/dropout/Mulhu  ČB
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/mul_3hu  ČB
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b,gradient_tape/sequential/dropout/dropout/Mulhu  ČB
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/mul_4hu  ČB
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/mul_5hu  ČB
W
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H bAdam/Adam/update/Sqrthu  ČB
K
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b
Adam/sub_1hu  ČB
K
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28 @ H b
Adam/sub_2hu  ČB
ą
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28 @ H b;gradient_tape/categorical_crossentropy/weighted_loss/Tile_1hu  ČB

Óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bdiv_no_nan_1hu  ČB
ŗ
Óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bEgradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nanhu  ČB
Ł
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28 @ H b*sequential/bidirectional/forward_gru/zeroshu  ČB
Š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *28 @ H bAdam/gradients/zeros_like_3hu  ČB
Ī
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *2(8 @ H bAdam/gradients/zeros_likehu  ČB
Š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *2(8 @ H bAdam/gradients/zeros_like_2hu  ČB
¤
ļvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8 @ H bAdam/gradients/AddNhu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAssignAddVariableOphu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAssignAddVariableOp_1hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAssignAddVariableOp_2hu  ČB
¬
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorReverseOp<Eigen::array<bool, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8 @ H b	ReverseV2hu  ČB

Łvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28 @ H bAdam/Adam/AssignAddVariableOphu  ČB
Ģ
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*2P8 @ H btranspose_0hu  ČB
÷
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorEvalToOp<Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::MakePointer> const, Eigen::GpuDevice>, long)*28 @ H b:categorical_crossentropy/softmax_cross_entropy_with_logitshu  ČB

=void tensorflow::SetToValue<float, float>(int, float*, float)*28 @ H b#Adam/Adam/update/UnsortedSegmentSumhu  ČB
z
Bvoid tensorflow::detail::RangeInitKernel<int>(int, int, int, int*)*28 @ H bAdam/Adam/update/Uniquehu  ČB

Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28 @ H b*categorical_crossentropy/weighted_loss/Sumhu  ČB
Ż
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type) *2 8 @ H bsequential/dense/Softmaxhu  ČB
ß
void tensorflow::scatter_op_gpu::ScatterOpCustomKernel<float, int, (tensorflow::scatter_op::UpdateOp)1>(float*, float const*, int const*, int, int, int)*28 @ H b%Adam/Adam/update/ResourceScatterAdd_1hu  ČB

xvoid tensorflow::unique_op_gpu::ExtractFirstOccurrenceIndicesKernel<int>(long, long, int const*, int const*, int*, int*)*28 @ H hu  ČB
¹
void tensorflow::unique_op_gpu::GatherOutputsAndInvertPermutationKernel<int, int>(long, int const*, int const*, int const*, int const*, int*, int*, int*) *28’@’H’hu  ČB
G
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*28@HbCast_1hu  ČB
`
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*28@Hbsequential/dropout/dropout/Casthu  ČB
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28@HbCast_2hu  ČB
z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*28@Hb8categorical_crossentropy/weighted_loss/num_elements/Casthu  ČB
M
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbAdam/truedivhu  ČB
`
!Equal_GPU_DT_INT64_DT_BOOL_kernel*28@Hbbatch_loss/write_summary/Equalhu  ČB
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*28@Hb
LogicalAndhu  ČB
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbAdam/Adam/update/mul_2hu  ČB
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbMulhu  ČB
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb
Adam/Pow_1hu  ČB
K
!Sqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb	Adam/Sqrthu  ČB
I
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28@HbAdam/subhu  ČB
K
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*28@Hb
Adam/sub_3hu  ČB
”
Óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@Hb,categorical_crossentropy/weighted_loss/valuehu  ČB
’
Óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@Hb
div_no_nanhu  ČB
Š
void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *28@HbAdam/gradients/zeros_like_1hu  ČB

ćvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28@HbAssignAddVariableOp_3hu  ČB

bvoid cub::DeviceScanInitKernel<cub::ScanTileState<int, true> >(cub::ScanTileState<int, true>, int)*28@HbAdam/Adam/update/Uniquehu  ČB

Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28@Hbsequential/dense/BiasAddhu  ČB
ė
Āvoid tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*28@HbSum_2hu  ČB
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*28’@’H’bAdam/Adam/update/mul_1hu  ČB

Łvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28’@’H’bAssignAddVariableOp_4hu  ČB
a
Bvoid tensorflow::detail::RangeInitKernel<int>(int, int, int, int*)*28@Hhu  ČB