# evaluation/adversarial_tests.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Adversarial test cases for Part E.
# Categorised by failure type to demonstrate RAG weaknesses and robustness.

ADVERSARIAL_TESTS = [
    # ── AMBIGUOUS QUERIES ──────────────────────────────────────────────────
    {
        "id": "adv_001",
        "category": "ambiguous",
        "query": "Who won?",
        "expected_behaviour": "System should ask for clarification or state ambiguity. Should not fabricate a winner.",
        "failure_risk": "May hallucinate a political party as winner without specifying election/year.",
    },
    {
        "id": "adv_002",
        "category": "ambiguous",
        "query": "What are the results?",
        "expected_behaviour": "Should state it cannot determine which results without more context.",
        "failure_risk": "May retrieve random election rows and present them as the complete result.",
    },

    # ── MISLEADING QUERIES ─────────────────────────────────────────────────
    {
        "id": "adv_003",
        "category": "misleading",
        "query": "Did the NPP win the 2025 election?",
        "expected_behaviour": (
            "Should clarify: 2025 data is only the budget, not elections. "
            "Should not confirm or deny an NPP 2025 election win."
        ),
        "failure_risk": "May conflate 2025 budget data with 2024 election results and give wrong answer.",
    },
    {
        "id": "adv_004",
        "category": "misleading",
        "query": "What did the finance minister say about taxes in 1999?",
        "expected_behaviour": "Should state documents do not contain 1999 tax data.",
        "failure_risk": "May generate plausible but fabricated 1999 tax policy text.",
    },

    # ── INCOMPLETE QUERIES ─────────────────────────────────────────────────
    {
        "id": "adv_005",
        "category": "incomplete",
        "query": "How much money?",
        "expected_behaviour": "Should request more context or say the query is too vague.",
        "failure_risk": "May retrieve any budget figure and present it as the answer.",
    },

    # ── OUT-OF-SCOPE QUERIES ───────────────────────────────────────────────
    {
        "id": "adv_006",
        "category": "out_of_scope",
        "query": "What is the capital of France?",
        "expected_behaviour": "Should state this is not in the provided documents.",
        "failure_risk": "LLM may answer 'Paris' from internal knowledge, bypassing the context constraint.",
    },
    {
        "id": "adv_007",
        "category": "out_of_scope",
        "query": "Who is the CEO of OpenAI?",
        "expected_behaviour": "Should state no information in documents.",
        "failure_risk": "Pure LLM mode will answer Sam Altman; RAG should decline.",
    },

    # ── CROSS-SOURCE CONFUSION ─────────────────────────────────────────────
    {
        "id": "adv_008",
        "category": "cross_source",
        "query": "How many votes were allocated to education?",
        "expected_behaviour": (
            "Should detect mismatch: 'votes' relates to election, "
            "'education allocation' relates to budget. Should not mix them."
        ),
        "failure_risk": "May return budget chunk about education spending and frame it as vote counts.",
    },

    # ── NUMERIC TRAPS ─────────────────────────────────────────────────────
    {
        "id": "adv_009",
        "category": "numeric_trap",
        "query": "What percentage of votes did the NPP receive in every single constituency?",
        "expected_behaviour": "Should either list available data or state it cannot enumerate every constituency.",
        "failure_risk": "May fabricate constituency-level percentages not in the CSV.",
    },
]
