# Visual extraction invariants

## INV-001 — The source inventory is exact

- Requirement: The destination contains exactly 27 reviewed families, 108 Cargo source packages, and 54 Bun source packages from the frozen extraction commit.
- Forbidden behavior: broad-copying ComfyUI or spatial-owned package families.
- Authority/source: issue:#2
- Affected surfaces: crates/**, packages/**, docs/repository-split/**
- Required evidence: static
- Sensitivity: optional
- Risk dimensions: security=not-applicable:no-security-change; recovery=covered:INV-003; persistence=not-applicable:no-persistence; concurrency=not-applicable:no-runtime; migration=covered:INV-003; partial-failure=covered:INV-003; operational=not-applicable:no-runtime

## INV-002 — Cross-repository boundaries are immutable

- Requirement: External Rust dependencies use exact registry coordinates; spatial reverse edges, monolith test support, sibling paths, and moving Git dependencies are absent.
- Forbidden behavior: hiding an absent prerequisite behind a local path or Git fallback.
- Authority/source: issue:#2
- Affected surfaces: Cargo.toml, crates/**/Cargo.toml
- Required evidence: static
- Sensitivity: optional
- Risk dimensions: security=not-applicable:no-security-change; recovery=covered:INV-003; persistence=not-applicable:no-persistence; concurrency=not-applicable:no-runtime; migration=covered:INV-003; partial-failure=covered:INV-003; operational=not-applicable:no-runtime

## INV-003 — Bootstrap provenance is recoverable

- Requirement: Unadapted source files retain byte identity and every adaptation is separated from the frozen source record; rust-packages remains untouched.
- Forbidden behavior: claiming behavioral verification or publication authority from structural extraction evidence.
- Authority/source: issue:#2
- Affected surfaces: docs/**, scripts/**, crates/**, packages/**
- Required evidence: contract
- Sensitivity: optional
- Risk dimensions: security=not-applicable:no-security-change; recovery=covered:INV-003; persistence=not-applicable:no-persistence; concurrency=not-applicable:no-runtime; migration=covered:INV-001; partial-failure=covered:INV-001; operational=not-applicable:no-runtime
