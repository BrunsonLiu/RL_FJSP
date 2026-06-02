# FJSP Instances

## Included Sets

- `tiny_2x2.fjs`: local toy instance for smoke tests.
- `brandimarte/`: 15 Brandimarte MK instances copied from the public `SchedulingLab/fjsp-instances` repository.
- `instances.json`: metadata for public benchmark instances, including known optima or bounds when available.

## Format Notes

Most public files use:

```text
<jobs> <machines>
<operation_count> <option_count> <machine> <processing_time> ...
```

Machine ids may be 0-based in public benchmark files and 1-based in some older examples. The repository parser normalizes both to 0-based ids internally.

## Source

Brandimarte instances are commonly used FJSP benchmarks from:

P. Brandimarte, "Routing and Scheduling in a Flexible Job Shop by Tabu Search", Annals of Operations Research, 1993.

The local copies were taken from:

https://github.com/SchedulingLab/fjsp-instances
