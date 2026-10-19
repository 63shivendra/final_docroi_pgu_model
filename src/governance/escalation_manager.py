"""
DocROI-Former: Human-in-the-Loop Governed Escalation Engine
===========================================================
Defines the three-tier triage logic for merchant onboarding documents:
  1. Straight-Through Processing (STP) Auto-Approval (P_doc < 0.15)
  2. Automatic Immediate Rejection (P_doc > 0.85 with verified tampering evidence)
  3. Governed Human Escalation (0.15 <= P_doc <= 0.85 or borderline evidence)

Design Pattern: State / Policy Strategy (SOLID: Open/Closed Principle)
"""

from typing import Dict, Any, Tuple
from src.core.reason_codes import GovernanceDecision


class GovernedEscalationManager:
    """
    Evaluates policy rules, risk confidence, and evidence sufficiency
    to assign operational triage routing.
    """

    def __init__(
        self,
        auto_approval_threshold: float = 0.15,
        rejection_threshold: float = 0.85,
        min_evidence_pixels: int = 35
    ):
        self.auto_approval_threshold = auto_approval_threshold
        self.rejection_threshold = rejection_threshold
        self.min_evidence_pixels = min_evidence_pixels

    def evaluate_triage(
        self,
        risk_score: float,
        tampered_pixels: int,
        has_font_inconsistency: bool = False
    ) -> Tuple[GovernanceDecision, str]:
        """
        Determines routing decision and generates operational summary explanation.
        """
        if risk_score < self.auto_approval_threshold and tampered_pixels < self.min_evidence_pixels:
            return GovernanceDecision.AUTO_APPROVED, "Clean authentic document. Approved for instant STP onboarding."

        if risk_score >= self.rejection_threshold and (tampered_pixels >= self.min_evidence_pixels or has_font_inconsistency):
            return GovernanceDecision.AUTO_REJECTED, "High-confidence forgery detected in legal entity zone. Rejected."

        return GovernanceDecision.ESCALATE_TO_HUMAN, "Borderline risk signals detected. Escalated to compliance review queue."
