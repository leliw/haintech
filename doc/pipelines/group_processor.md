# GroupProcessor

Groups the data based on the expression or key
and returns aggregated data.

Type parameters:

* I - type of input items
* K - type of key value
* O - type of output items

## Constructor

Arguments:

* group_by: The expression to group by, gets input data and returns the key
* init_group: The function to initialize the group, gets the key and the input data and returns the group (output data)
* aggregate: The function to aggregate the data to the group, gets the group (output data) and the input data and returns the group (output data) or None

## Use cases

Below input stream is grouped by "i" property, each group is initialised with dictionary where
"l" property is empty list. While grouping (aggregating) each "l" property is added to list.

```python
stream = [
    {"i": 0, "l": 1},
    {"i": 0, "l": 2},
    {"i": 0, "l": 3},
    {"i": 1, "l": 4},
    {"i": 1, "l": 5},
    {"i": 1, "l": 6},
]
pl = Pipeline(
    [
        GroupProcessor(
            group_by=lambda x: x["i"],
            init_group=lambda k, d: {"i": k, "l": []},
            aggregate=lambda g, k, d: g["l"].append(d["l"]),
        )
    ]
)
ret = await pl.run_and_return(stream)
assert [{"i": 0, "l": [1, 2, 3]}, {"i": 1, "l": [4, 5, 6]}] == ret
```
