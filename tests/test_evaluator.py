"""Unit tests for the Exact Amazon-Style Macro F_0.5 Evaluator.

Verifies test cases A through J as explicitly required by Phase 0 specification,
as well as the exact numerical example from the Amazon problem statement.
"""

import pytest
from src.evaluation.metric import evaluate_entity_f05, evaluate_f05, EvaluationResult


def test_challenge_problem_statement_example():
    """Reproduce the official numerical example from the challenge document.

    Example from README / problem statement:
    - Model predicts: S1-00001 -> [S2-00047, S2-00193, S3-00812]
    - Ground truth:  S1-00001 -> [S2-00047, S3-00812]
    - TP = 2
    - Precision = 2/3 ≈ 0.666667
    - Recall = 2/2 = 1.0
    - Numerator = 1.25 * (2/3) * 1.0 = 5/6
    - Denominator = 0.25 * (2/3) + 1.0 = 7/6
    - F_0.5 = (5/6) / (7/6) = 5/7 ≈ 0.714285714... (rounds to 0.714)
    """
    true_ids = ["S2-00047", "S3-00812"]
    pred_ids = ["S2-00047", "S2-00193", "S3-00812"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)

    assert prec == pytest.approx(2.0 / 3.0)
    assert rec == pytest.approx(1.0)
    # Exact fraction 5/7
    assert f05 == pytest.approx(5.0 / 7.0)
    assert round(f05, 3) == 0.714


def test_case_a_perfect_one_match():
    """Case A: Perfect one-match prediction."""
    true_ids = ["S2-00001"]
    pred_ids = ["S2-00001"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert f05 == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_case_b_completely_empty_prediction():
    """Case B: Completely empty prediction when true match exists."""
    true_ids = ["S2-00001"]
    pred_ids = []

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_case_c_one_missed_true_match():
    """Case C: One missed true match (2 true, 1 predicted).

    True: [S2-1, S3-1], Pred: [S2-1]
    TP = 1, Prec = 1/1 = 1.0, Rec = 1/2 = 0.5
    Num = 1.25 * 1.0 * 0.5 = 0.625
    Den = 0.25 * 1.0 + 0.5 = 0.75
    F0.5 = 0.625 / 0.75 = 5/6 ≈ 0.8333333333333334
    """
    true_ids = ["S2-00001", "S3-00001"]
    pred_ids = ["S2-00001"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert prec == 1.0
    assert rec == 0.5
    assert f05 == pytest.approx(5.0 / 6.0)


def test_case_d_one_false_positive():
    """Case D: One false positive (1 true, 2 predicted).

    True: [S2-1], Pred: [S2-1, S3-2]
    TP = 1, Prec = 1/2 = 0.5, Rec = 1/1 = 1.0
    Num = 1.25 * 0.5 * 1.0 = 0.625
    Den = 0.25 * 0.5 + 1.0 = 1.125
    F0.5 = 0.625 / 1.125 = 5/9 ≈ 0.5555555555555556
    """
    true_ids = ["S2-00001"]
    pred_ids = ["S2-00001", "S3-00002"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert prec == 0.5
    assert rec == 1.0
    assert f05 == pytest.approx(5.0 / 9.0)


def test_case_e_partial_multi_match():
    """Case E: Partial multi-match prediction (3 true, 2 predicted, 1 correct, 1 FP).

    True: [S2-1, S2-2, S3-1], Pred: [S2-1, S3-99]
    TP = 1, Prec = 1/2 = 0.5, Rec = 1/3 ≈ 0.333333
    Num = 1.25 * 0.5 * (1/3) = 1.25 / 6 = 5/24
    Den = 0.25 * 0.5 + (1/3) = 1/8 + 1/3 = 11/24
    F0.5 = (5/24) / (11/24) = 5/11 ≈ 0.45454545454545453
    """
    true_ids = ["S2-00001", "S2-00002", "S3-00001"]
    pred_ids = ["S2-00001", "S3-00099"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert prec == 0.5
    assert rec == pytest.approx(1.0 / 3.0)
    assert f05 == pytest.approx(5.0 / 11.0)


def test_case_f_completely_incorrect_multi_match():
    """Case F: Completely incorrect multi-match prediction."""
    true_ids = ["S2-00001", "S2-00002"]
    pred_ids = ["S3-00001", "S3-00002"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_case_g_correct_singleton():
    """Case G: Correct singleton (True empty, predicted empty => 1.0 credit)."""
    true_ids = []
    pred_ids = []

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert f05 == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_case_h_false_match_on_singleton():
    """Case H: False match on singleton (True empty, predicted match => 0.0 penalty)."""
    true_ids = []
    pred_ids = ["S2-00001"]

    f05, prec, rec = evaluate_entity_f05(true_ids, pred_ids)
    assert f05 == 0.0
    assert prec == 0.0
    assert rec == 0.0


def test_case_i_duplicate_predicted_id_rejection():
    """Case I: Duplicate predicted ID inside list raises ValueError."""
    true_ids = ["S2-00001"]
    pred_ids = ["S2-00001", "S2-00001"]

    with pytest.raises(ValueError, match="Duplicate predicted entity IDs detected"):
        evaluate_entity_f05(true_ids, pred_ids, check_duplicates=True)


def test_case_j_multiple_s1_entities_macro_average():
    """Case J: Multiple S1 entities macro-averaging and full diagnostics."""
    gt = {
        "S1-1": ["S2-1"],         # Case A: F0.5 = 1.0
        "S1-2": [],               # Case G (correct singleton): F0.5 = 1.0
        "S1-3": [],               # Case H (false merge singleton): F0.5 = 0.0
        "S1-4": ["S2-2"],         # Case B (missed non-singleton): F0.5 = 0.0
    }

    preds = {
        "S1-1": ["S2-1"],
        "S1-2": [],
        "S1-3": ["S3-9"],
        "S1-4": [],
    }

    res = evaluate_f05(gt, preds)

    # Expected Macro F0.5 = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.50
    assert res.macro_f05 == pytest.approx(0.50)
    assert res.num_s1_entities == 4
    assert res.num_true_singletons == 2
    assert res.num_correct_singletons == 1
    assert res.num_false_singleton_merges == 1
    assert res.num_entities_with_predictions == 2
    assert res.total_true_links == 2
    assert res.total_predicted_links == 2
    assert res.mean_true_links_per_s1 == 0.5
    assert res.mean_predicted_links_per_s1 == 0.5


def test_missing_entity_in_predictions_raises_key_error():
    """Missing required S1 entities from predictions must raise KeyError under strict mode."""
    gt = {"S1-1": ["S2-1"], "S1-2": []}
    preds = {"S1-1": ["S2-1"]}  # missing S1-2

    with pytest.raises(KeyError, match="Predictions missing 1 required Source-1 entities"):
        evaluate_f05(gt, preds, strict_entity_coverage=True)
