# Limit

Simply limits numer of processed items by pipeline.
It is usefull while debugging.

You can it intialize with one number (pass only first X items) or
range (pass items from X to Y),

## Usage

```python
pl = Pipeline[str, str]([Limit(2)])
ret = await pl.run_and_return(["a", "b", "c"])
assert ret == ["a", "b"]
```

```python
pl = Pipeline[str, str]([Limit(1,3)])
ret = await pl.run_and_return(["a", "b", "c", "d", "e", "f"])
assert ret == ["b", "c", "d"]
```
