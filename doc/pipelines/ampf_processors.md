# AMPF processors

AMPF processors are wrappers of AMPF classes and methods.

## BlobStorageReader

Downloads blob from storage. Gets key walue add returns blob (bytes or str) and metadata.

```python
    # Given: Storage with uploaded blob
    storage = factory.create_blob_storage("test", D, "text/plain")
    storage.upload_blob(1, b"test", metadata)
    # And: Pipeline with BlobStorageReader
    pl = Pipeline(
        [
            BlobStorageReader(storage),
        ]
    )
    # When: Run pipeline with key of stored blob
    ret = await pl.run_and_return(1)
    # Then: Returns blom and metadata
    assert ret[0] == "test"
    assert ret[1] == metadata
```

## BlobStorageWriter

Uploads blob to storage. Gets blob (bytes or str) and metadata and returs the same.

Key_name parameter specifies how to obtain key value (file name) from metadata.
It can be metadatas attribute name or lambda function that returns key value.

```python
    # Given: Storage
    storage = factory.create_blob_storage("test", D, "text/plain")
    # And: Pipeline with BlobStorageWriter 
    pl = Pipeline(
        [
            BlobStorageWriter(storage, key_name="page_no"),
        ]
    )
    # When: Run pipeline with blob and metadata
    await pl.run_and_return(("test", metadata))
    # Then: Blob and metadata are uploaded
    keys = list(storage.keys())
    assert len(keys) == 1
    assert keys[0] == 1
    assert storage.download_blob(keys[0]) == b"test"
    assert storage.get_metadata(keys[0]) == metadata
```
