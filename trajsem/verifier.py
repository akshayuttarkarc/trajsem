from __future__ import annotations

import re
from trajsem.schemas import SemanticEvent, VerificationFinding

_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+\s*(?:Å|A|ns|%|frames?|Å³|A3)?", re.I)
_TAG_RE = re.compile(r"\[E(\d+)\]")


def split_claims(markdown: str) -> list[str]:
    text = re.sub(r"```.*?```", "", markdown, flags=re.S)
    text = re.sub(r"([.!?])\s+(\[E\d+\])", r" \2\1", text)
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [c.strip() for c in candidates if len(c.strip()) > 30 and not c.strip().startswith("#")]


def verify_report(markdown: str, events: list[SemanticEvent]) -> list[VerificationFinding]:
    findings: list[VerificationFinding] = []
    for claim in split_claims(markdown):
        tags = [int(x) for x in _TAG_RE.findall(claim)]
        supporting = []
        for tag in tags:
            if 1 <= tag <= len(events):
                supporting.append(events[tag - 1].title)
        has_specific = bool(_NUMERIC_RE.search(claim)) or any(w in claim.lower() for w in ["hydrogen", "pocket", "ligand", "contact", "rmsd", "salt", "loop"])
        risky = any(w in claim.lower() for w in ["affinity", "thermodynamic", "causes", "proves", "allosteric pathway", "binding free energy"])
        if risky and not supporting:
            status = "unsupported_risky_claim"
            reason = "Risky mechanistic or thermodynamic language appears without an evidence tag."
        elif has_specific and not supporting:
            status = "needs_evidence_tag"
            reason = "Specific MD claim appears without [E#] evidence grounding."
        elif supporting:
            status = "supported_by_event_tag"
            reason = "Claim contains at least one valid event evidence tag."
        else:
            status = "general_text"
            reason = "General framing or limitation text."
        findings.append(VerificationFinding(claim=claim, status=status, supporting_events=supporting, reason=reason))
    return findings


def verification_summary(findings: list[VerificationFinding]) -> dict:
    total = len(findings)
    bad = sum(f.status in {"unsupported_risky_claim", "needs_evidence_tag"} for f in findings)
    supported = sum(f.status == "supported_by_event_tag" for f in findings)
    return {
        "claims_checked": total,
        "supported_claims": supported,
        "flagged_claims": bad,
        "unsupported_rate": round(bad / total, 4) if total else 0.0,
    }
