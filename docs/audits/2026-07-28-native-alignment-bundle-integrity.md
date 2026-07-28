# Native alignment import bundle — integrity diagnosis

**Date:** 2026-07-28  
**Branch:** `agent/alignment-tolerant-v1`

The previously committed transport file `.github/import/native-alignment-v1.b64` was inspected before reuse, as required by the research handoff.

Expected decoded archive SHA-256 from the import workflow:

```text
9e57be34c7d67cc5a90684be8eb3798be4de4b350f9675f9cb1b8039cd8bec8c
```

Observed decoded byte-stream SHA-256 from the exact Git blob:

```text
310f6917b2060390127ccc119f9c0b061390fe06193e0cf93581d659dee06763
```

`xz` integrity validation also reports corrupted compressed data. Therefore the transport bundle cannot be treated as a valid scientific source or extracted implementation. Retriggering its workflow cannot repair the underlying bytes.

Decision:

```text
CORRUPT_TRANSPORT_BUNDLE_REJECTED
```

The import workflow and corrupt payload are removed from the active experiment branch. The alignment-tolerant implementation is recreated as readable, tested source, with a fresh preregistration and independent audit path.
