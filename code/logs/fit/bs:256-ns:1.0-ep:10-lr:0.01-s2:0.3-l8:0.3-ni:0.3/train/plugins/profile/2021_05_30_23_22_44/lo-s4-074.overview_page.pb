?	!?jK?@!?jK?@!!?jK?@	Z?Oz?G@Z?Oz?G@!Z?Oz?G@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?jK?@o??=???A??y ?H?@Y,??y?J?@*	y?&???%A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchn??4J?@!??t??X@)n??4J?@1??t??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismr???UJ?@!?П??X@){?%T??1,@???Pb?:Preprocessing2F
Iterator::Model?i?aJ?@!      Y@)5)?^?x?1??^??K?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 47.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9Y?Oz?G@I?]J@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	o??=???o??=???!o??=???      ??!       "      ??!       *      ??!       2	??y ?H?@??y ?H?@!??y ?H?@:      ??!       B      ??!       J	,??y?J?@,??y?J?@!,??y?J?@R      ??!       Z	,??y?J?@,??y?J?@!,??y?J?@b      ??!       JCPU_ONLYYY?Oz?G@b q?]J@Y      Y@q???cr?"?
host?Your program is HIGHLY input-bound because 47.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 