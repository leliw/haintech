# Pipeline

Pipeline class that runs a series of processors on data.

## Constructor

Just pass an arrray with processors.

```python
    pl = Pipeline(
        [
            LambdaProcessor[int, int](lambda x: x + 1, name="L1"),
            FakeCheckpoint[int](),
            LambdaProcessor[int, int](lambda x: x + 2, name="L2"),
        ]
    )
```

## Methods

### run()

Run pipeline with data. Return async generator of results.

Args:

* data: Data to process which is sent to the first processor.

### run_and_return()

Run pipeline and return result.
If the last processor returns a generator, return a list of results.

### get_step()

Return part (step) of pipeline. Pipeline is divided into steps by CheckpointProcessor.
This method returns a new pipeline with processors from the step.

Arguments:

* no: Number of step.

Returns:

* Pipeline: New pipeline with processors from the step.

This method is usefull for tests. You can divide big pipeline into
steps and test each step separately.

```python
storage = factory.create_storage("test3", D, "content")
pl = Pipeline(
    [
        LambdaProcessor[D, D](lambda d: setattr(d, "page_no", 2)),
        StorageWriter(storage),
        LambdaProcessor[D, D](lambda d: setattr(d, "content", "2")),
    ]
)

pl0 = pl.get_step(0)
pl1 = pl.get_step(1)

await pl0.run_and_return(data)

ret = await pl1.run_and_return(storage.keys())
assert ret[0].page_no == 2
assert ret[0].content == "2"
```
