	? ?> QO@? ?> QO@!? ?> QO@	؅f???P@؅f???P@!؅f???P@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$? ?> QO@,g~5??A?P??4@Y?yȔ	E@*	L7?A ??@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???4?D@!/?jF?X@)???4?D@1/?jF?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?~?fE@!????X@)??"ڎ??1Q??i??:Preprocessing2F
Iterator::Model??qQ-E@!      Y@)?4Lk?x?1b?銍?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 67.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9؅f???P@IP?2?,j@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,g~5??,g~5??!,g~5??      ??!       "      ??!       *      ??!       2	?P??4@?P??4@!?P??4@:      ??!       B      ??!       J	?yȔ	E@?yȔ	E@!?yȔ	E@R      ??!       Z	?yȔ	E@?yȔ	E@!?yȔ	E@b      ??!       JCPU_ONLYY؅f???P@b qP?2?,j@@