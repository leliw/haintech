# ProgressTracker

Tracks the progress of a pipeline.

## Usage

```python
pt = ProgressTracker()
pt.set_total_steps(len(sitemap.urlset))
for ret in sitemap.urlset:
   pt.increment()
```
