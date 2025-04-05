# LambdaProcessor

The LambdaProcessor just run given lambda expression on all items in pipeline.

```python
pl = Pipeline([LambdaProcessor[int, int](lambda x: x + 1)])
ret = await pl.run_and_return(5)
assert ret == 6
```

If expression doesn't return value, the input item is returned.

```python
pl = Pipeline([LambdaProcessor[int, int](lambda x: print(x + 1))])
ret = await pl.run_and_return(5)
assert ret == 5
```

This feature is particularly useful when working with Pydantic objects.
The setattr() function allows you to change an object's attribute value and return the updated object.

```python
class D(BaseModel):
    val: int

pl = Pipeline([LambdaProcessor[int, int](lambda d: setattr(d, "val", d.val + 1))])
ret = await pl.run_and_return(D(val=5))
assert ret.val == 6
```
