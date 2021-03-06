?	ő"? @ő"? @!ő"? @	?]6*?@?]6*?@!?]6*?@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9ő"? @S?!?uqk?1Z?xZ~`??A?? w???I?ю~7@Y??y?Cn??r0*	4^?I??@2i
2Iterator::Root::Prefetch::MemoryCacheImpl::BatchV2????9??!9? I??W@)?q6??1??Q?B@:Preprocessing2s
;Iterator::Root::Prefetch::MemoryCacheImpl::BatchV2::Shuffle??b9??!`?????K@)7?~T??1???r??=@:Preprocessing2?
HIterator::Root::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::TensorSlice???^(`;??!?3????9@)??^(`;??1?3????9@:Preprocessing2E
Iterator::Root?x@ٔ+??!Ǟt?:@)??'d?m??1s???F@:Preprocessing2O
Iterator::Root::Prefetchg)YNB???!??d??@)g)YNB???1??d??@:Preprocessing2`
)Iterator::Root::Prefetch::MemoryCacheImpl?/g?+t??!%?? ?qW@)?N??C}?1?u`?????:Preprocessing2\
%Iterator::Root::Prefetch::MemoryCache,cC7????!?8PW@);?f??_?1!?M?? ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 2.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?75.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?]6*?@I?
)* S@Q?f~=5@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	S?!?uqk?S?!?uqk?!S?!?uqk?      ??!       "	Z?xZ~`??Z?xZ~`??!Z?xZ~`??*      ??!       2	?? w????? w???!?? w???:	?ю~7@?ю~7@!?ю~7@B      ??!       J	??y?Cn????y?Cn??!??y?Cn??R      ??!       Z	??y?Cn????y?Cn??!??y?Cn??b      ??!       JGPUY?]6*?@b q?
)* S@y?f~=5@?"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop?L???I??!?L???I??0"&
CudnnRNNCudnnRNN?=7???!??gT????"&
concat_0ConcatV2???[ҥ?!?5??D??"m
:categorical_crossentropy/softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogitsOr???ѥ?!?l????"3
Adam/Adam/update/UniqueUnique(?????!?SZ?????";
gradients/split_2_grad/concatConcatV2??i"???!??mL?<??"%
Adam/Cast_1Cast???$???!x?+?????";
gradients/split_1_grad/concatConcatV2hJͦo[??!?(ǝj???"9
gradients/split_grad/concatConcatV2???? ??!@???T??"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad?????p??!;?M?????Q      Y@Y?O?v?J@at?f??)G@q???/e, @y{n?d???"?

device?Your program is NOT input-bound because only 2.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?75.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Volta)(: B 