"""Label set for insurance sentence classification. Single source of truth."""

from __future__ import annotations

LABELS: list[str] = [
    "coverage_grant",
    "exclusion",
    "premium_payment",
    "deductible_limits",
    "claims_procedure",
    "definitions",
    "term_cancellation_renewal",
    "conditions_duties",
]

DESCRIPTIONS: dict[str, str] = {
    "coverage_grant": (
        "States what the policy covers or insures: the insuring agreement, covered perils, "
        "covered persons/property, benefits payable."
    ),
    "exclusion": (
        "States what is NOT covered: excluded perils, activities, property, circumstances, "
        "or losses the insurer will not pay for."
    ),
    "premium_payment": (
        "Concerns the premium: amount, due dates, payment methods, grace periods, "
        "late fees, premium adjustments or refunds."
    ),
    "deductible_limits": (
        "Concerns deductibles, co-payments, coinsurance, sub-limits, policy limits, "
        "caps, or maximum amounts payable."
    ),
    "claims_procedure": (
        "Describes how to file, document, or process a claim: notice of loss, proof of loss, "
        "claim forms, adjuster contact, timelines for claim submission or payment."
    ),
    "definitions": (
        "Defines a term used in the policy (e.g. 'Insured means...', 'Occurrence means...'), "
        "typically in a Definitions section."
    ),
    "term_cancellation_renewal": (
        "Concerns the policy period, effective/expiration dates, cancellation by either party, "
        "non-renewal, renewal terms, or reinstatement."
    ),
    "conditions_duties": (
        "States duties or conditions of the insured or insurer other than claims filing: "
        "duty to cooperate, disclosure/misrepresentation, subrogation, other insurance, "
        "inspections, changes in risk, assignment, arbitration/appraisal."
    ),
}

LABEL2ID: dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


def label_block() -> str:
    """Markdown block of labels + descriptions, used verbatim in prompts."""
    lines = []
    for i, l in enumerate(LABELS):
        lines.append(f"{i}. `{l}` — {DESCRIPTIONS[l]}")
    return "\n".join(lines)
