# Haintech

A utility package for AI solutions.
There are two packages:

* [ai](doc/ai.md) - abstraction layer and utilities for LLMs
* pipelines - framework for buiding pipelines

## Build and publish

```bash
export ARTIFACT_REGISTRY_TOKEN=$(
    gcloud auth application-default print-access-token
)
export UV_PUBLISH_USERNAME=oauth2accesstoken
export UV_PUBLISH_PASSWORD="$ARTIFACT_REGISTRY_TOKEN"

rm -rf dist/
uv build
uv publish --index private-registry
```
