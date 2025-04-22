# Pipelines Package  

A collection of classes that enable the construction of pipelines.  
A pipeline allows for sequential data processing through a series of processors.  

* [Pipeline](pipelines/pipeline.md) – The main class representing a pipeline.  
* [BaseProcessor](pipelines/base_processor.md) – The base class for all processors.  
* [ConcurrentProcessor](pipelines/concurrent_processor.md) – A processor that processes data concurrently.  

## Processor Classes  

* [LambdaProcessor](pipelines/lambda_processor.md) –
  a processor that performs an operation defined by a lambda expression.  
* [PipelineProcessor](pipelines/pipeline_processor.md) –
  a processor that executes another pipeline.  
* [GroupProcessor](pipelines/group_processor.md) –
  a processor that groups data.  
* [BlobStorageReader](pipelines/ampf_processors.md) –
  a processor that retrieves a blob from storage.  
* [BlobStorageWriter](pipelines/ampf_processors.md) –
  a processor that saves a blob to storage.

## Helper Classes

* [ProgressTracker](pipelines/progress_tracker.md) –
  tracks the progress of a pipeline.
