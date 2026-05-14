# Research Version Implementation Plan

## Claim to defend

TrajSem does not claim that an LLM directly understands raw MD trajectories. The defensible claim is narrower: structured trajectory descriptors can be transformed into auditable semantic events, and a constrained language model can generate faithful scientific reports from those events.

## Core modules

1. Raw descriptor extraction through MDAnalysis.
2. Residue-level interaction geometry.
3. Pocket-volume integration via CSV import and optional fpocket adapter.
4. Semantic event ontology.
5. Grounded report generation.
6. Hallucination/claim verifier.
7. Expert annotation and evaluation workflow.

## Paper-grade evaluation

Use expert annotations and compare against:

1. Rule-only reporting.
2. Direct LLM prompting over descriptor CSV.
3. Event-grounded LLM reporting.
4. Human-written expert summaries.

Report event precision, recall, F1, temporal IoU, claim support rate, unsupported-claim rate, and blinded expert usefulness scores.
