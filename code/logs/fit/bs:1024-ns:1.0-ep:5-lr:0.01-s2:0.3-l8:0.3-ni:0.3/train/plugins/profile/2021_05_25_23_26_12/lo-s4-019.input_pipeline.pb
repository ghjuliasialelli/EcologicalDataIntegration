	???v?j]@???v?j]@!???v?j]@	?y???T@?y???T@!?y???T@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???v?j]@???P???A????*(3@YDԷ̉X@*	a??"???@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??>U??X@!??S?X@)??>U??X@1??S?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismfh<ĆX@!??????X@)h׿???1???h?M??:Preprocessing2F
Iterator::Model?:?*?X@!      Y@)Ҫ?t??y?1???Ez?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 83.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?y???T@I?????0@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???P??????P???!???P???      ??!       "      ??!       *      ??!       2	????*(3@????*(3@!????*(3@:      ??!       B      ??!       J	DԷ̉X@DԷ̉X@!DԷ̉X@R      ??!       Z	DԷ̉X@DԷ̉X@!DԷ̉X@b      ??!       JCPU_ONLYY?y???T@b q?????0@