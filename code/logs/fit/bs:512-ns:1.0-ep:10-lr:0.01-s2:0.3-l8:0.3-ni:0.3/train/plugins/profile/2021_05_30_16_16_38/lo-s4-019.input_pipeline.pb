	{?????@{?????@!{?????@	????Lm@????Lm@!????Lm@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails${?????@a?4????A???\???@Y?3M?~?\@*	_?I??@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch!?rh??\@!c?o??X@)!?rh??\@1c?o??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?̰Q??\@!o{$u??X@)????N??1???K륑?:Preprocessing2F
Iterator::ModelJ'L5?\@!      Y@)?z?V??w?1?4$ᶢt?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 4.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????Lm@IFƄ3+?W@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	a?4????a?4????!a?4????      ??!       "      ??!       *      ??!       2	???\???@???\???@!???\???@:      ??!       B      ??!       J	?3M?~?\@?3M?~?\@!?3M?~?\@R      ??!       Z	?3M?~?\@?3M?~?\@!?3M?~?\@b      ??!       JCPU_ONLYY????Lm@b qFƄ3+?W@