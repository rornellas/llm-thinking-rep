# Audited pre-Qwen research bundle

This branch contains a hash-verified export of the audited pre-Qwen certification, fail analysis, methodological claim standard, preregistrations, tests, and experiment sources.

## Archive integrity

Concatenate the numbered Base64 parts in lexical order and decode them:

```bash
cat .github/import/preqwen-latest/part-*.b64 | base64 -d > /tmp/preqwen-latest.tar.xz
sha256sum /tmp/preqwen-latest.tar.xz
```

Expected SHA-256:

```text
3b53c763142b8f0cbaadee11a5fad0ece24a0a24fa32dabc777f6dc1793a0143
```

Extract at the repository root:

```bash
tar -xJf /tmp/preqwen-latest.tar.xz -C .
python -m compileall -q pre_qwen_certification tests
python -m pytest -q
sha256sum -c LATEST_RESEARCH_MANIFEST.sha256
```

The temporary workflow `_import-preqwen-latest.yml` performs the same reconstruction and validation. The source package was frozen before the follow-up experiment runs. Important claims are governed by `docs/methodology/IMPORTANT_CLAIM_VERIFICATION_STANDARD.md` and require factual recomputation plus an independent adversarial audit.
