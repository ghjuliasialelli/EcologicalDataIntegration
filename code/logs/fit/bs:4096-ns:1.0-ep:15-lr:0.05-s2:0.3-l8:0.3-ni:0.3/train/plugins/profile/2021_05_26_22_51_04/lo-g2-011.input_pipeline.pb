	X??,??@X??,??@!X??,??@	?
?_??J??
?_??J?!?
?_??J?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X??,??@???3ڪ??Af??????@Y??*????*	X9??jPA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ޫZϰ@!"{؈??X@)??ޫZϰ@1"{؈??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??0?њ?!1d????C?)??0?њ?11d????C?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??\???!?ʫ??JP?)?d??)??1?b>??I9?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??@]ϰ@!???X@)?-</??1?JO?i,?:Preprocessing2F
Iterator::Model?;l"3??!????Q?)??+?pq?1?ԂT??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?
?_??J?I????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???3ڪ?????3ڪ??!???3ڪ??      ??!       "      ??!       *      ??!       2	f??????@f??????@!f??????@:      ??!       B      ??!       J	??*??????*????!??*????R      ??!       Z	??*??????*????!??*????b      ??!       JCPU_ONLYY?
?_??J?b q????X@