	??
aE??@??
aE??@!??
aE??@	?ey
?I@?ey
?I@!?ey
?I@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??
aE??@??_ ??A???J㓇@Y?ׁs濇@*	?~j?+1'A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch	Q????@!?????X@)	Q????@1?????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismN?????@!?????X@)??F??R??1??┞?_?:Preprocessing2F
Iterator::Model??????@!      Y@)5????Ku?1-???jF?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 50.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?ey
?I@Is???5?H@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??_ ????_ ??!??_ ??      ??!       "      ??!       *      ??!       2	???J㓇@???J㓇@!???J㓇@:      ??!       B      ??!       J	?ׁs濇@?ׁs濇@!?ׁs濇@R      ??!       Z	?ׁs濇@?ׁs濇@!?ׁs濇@b      ??!       JCPU_ONLYY?ey
?I@b qs???5?H@