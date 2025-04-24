# FlatMapProcessor

The FlatMapProcessor is a processor that flattens a list of lists into a single list.
It is similar to the `flatMap` operation in functional programming.

## Constructor

Arguments:

* iterable: Callable[[I], Iterable[O]] = None
* output: FieldNameOrLambda = None,

## Usage

Simple use without any argument.

```python
pl = Pipeline([FlatMapProcessor[List[int], int]()])
ret = await pl.run_and_return([[1, 2], [3, 4]])
assert ret == [1, 2, 3, 4]
```

Using iterable parameter.

```python
class D(BaseModel):
    page_no: int
    subitems: List[str]

pl = Pipeline[D, str](
    [
        FlatMapProcessor[D, str](lambda d: d.subitems),
    ]
)
ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
assert ret == ["a", "b"]
```

Using iterable and output parameters.

```python
pl = Pipeline[D, str](
    [
        FlatMapProcessor[D, str](
            iterable=lambda d: d.subitems,
            output=lambda d, r: r + "x",
        ),
    ]
)
ret = await pl.run_and_return(D(page_no=1, subitems=["a", "b"]))
assert ret == ["ax", "bx"]
```
