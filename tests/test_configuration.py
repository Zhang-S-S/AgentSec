from __future__ import annotations

import unittest

from sara.experiment import VARIANTS
from multisource.model_risk_residual_mil import RiskResidualMILConfig


class ConfigurationTest(unittest.TestCase):
    def test_paper_defaults(self) -> None:
        config = RiskResidualMILConfig(
            embedding_dim=4096,
            use_mil_branch=True,
            behavior_encoder=False,
            post_top_k=3,
            comment_top_k=5,
        )
        config.validate()
        self.assertEqual(config.post_top_k, 3)
        self.assertEqual(config.comment_top_k, 5)

    def test_ablation_variants(self) -> None:
        self.assertIn("full", VARIANTS)
        self.assertIn("pure_mil", VARIANTS)
        self.assertIn("without_graph", VARIANTS)


if __name__ == "__main__":
    unittest.main()
