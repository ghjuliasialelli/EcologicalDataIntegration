	?J??x??@?J??x??@!?J??x??@	?&??O8???&??O8??!?&??O8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?J??x??@?3?{@A?0Bx ??@Y?]???@*	P??n?,?@2]
&Iterator::Model::MaxIntraOpParallelismr?t???@!???g?X@)????)n@1??E,$S@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??5Φc??!?W???[7@)??5Φc??1?W???[7@:Preprocessing2F
Iterator::Model?l#??@!      Y@)????>??1dk5:????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?&??O8??I? {l?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?3?{@?3?{@!?3?{@      ??!       "      ??!       *      ??!       2	?0Bx ??@?0Bx ??@!?0Bx ??@:      ??!       B      ??!       J	?]???@?]???@!?]???@R      ??!       Z	?]???@?]???@!?]???@b      ??!       JCPU_ONLYY?&??O8??b q? {l?X@