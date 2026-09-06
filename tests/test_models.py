import numpy as np
import pandas as pd
import pytest
import torch

from bioseq_models import available_models, load_model, model_info
from bioseq_models.tabular import CetnarLightGBM
from sequence_optimization import SequenceSpace, fast_seqprop_optimize


MODEL_CASES = [
    ("optimus5prime", "A" * 50, 1, {}),
    ("deepstarr", "A" * 249, 2, {}),
    ("dream_legnet", "A" * 110, 1, {}),
    ("cetnar_stability", "A" * 60, 2, {}),
    ("parm_k562", "A" * 80, 1, {}),
    ("agarwal_hepg2", "A" * 230, 1, {}),
    ("agarwal_k562", "A" * 230, 1, {}),
    ("agarwal_wtc11", "A" * 230, 1, {}),
    ("kosuri_composability", "A" * 109, 2, {}),
    ("zhao_ptrc", "A" * 74, 1, {}),
    ("zhao_ptrc_xgboost", "A" * 74, 1, {}),
    ("zhou_l3_conditional", "A" * 29, 1, {"sublibrary": "L3_s2"}),
    ("zhou_l3", "A" * 29, 1, {}),
    ("tfbind8_weak", "A" * 8, 1, {}),
    ("utr5_weak", "A" * 50, 1, {}),
    ("dream_weak", "A" * 110, 1, {}),
]


@pytest.mark.parametrize(("name", "sequence", "tasks", "options"), MODEL_CASES)
def test_bundled_model_loads_and_predicts(name, sequence, tasks, options):
    model = load_model(name, device="cpu", batch_size=2, **options)
    prediction = model.score([sequence]).scores
    expected_shapes = {(1, tasks)}
    if tasks == 1:
        expected_shapes.add((1,))
    assert prediction.shape in expected_shapes
    assert np.isfinite(prediction).all()
    assert model.task_names == model_info(name).task_names


def test_registry_and_aliases():
    assert available_models() == tuple(case[0] for case in MODEL_CASES)
    assert model_info("optimus").name == "optimus5prime"
    assert model_info("dream").sequence_length == 110
    assert model_info("tfbind8").status == "weak_surrogate"
    assert model_info("dream_legnet").differentiable
    assert model_info("kosuri").input_kind == "75nt_promoter+34nt_rbs"
    assert set(available_models(status="paper_architecture")) == {
        "optimus5prime",
        "deepstarr",
        "dream_legnet",
        "parm_k562",
        "agarwal_hepg2",
        "agarwal_k562",
        "agarwal_wtc11",
    }


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_model_loader_supports_variable_length_sequences(padding_side):
    model = load_model(
        "optimus5prime",
        device="cpu",
        pad_to_length=50,
        padding_side=padding_side,
    )
    prediction = model.score(["ACGT" * 10]).scores
    assert prediction.shape == (1, 1)
    assert np.isfinite(prediction).all()


def test_cetnar_exposes_decay_and_higher_is_better_stability():
    model = load_model("cetnar_stability", device="cpu", padding_side="left")
    values = model.score(["ACGT" * 10, "ACGT" * 15]).scores
    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[:, 0], -values[:, 1])


def test_kosuri_pair_api_matches_canonical_combined_slots():
    model = load_model("kosuri_composability", device="cpu")
    promoter, rbs = "ACGT" * 10, "TGCA" * 5
    paired = model.score_pairs([promoter], [rbs]).scores
    combined = promoter.ljust(75, "N") + rbs.ljust(34, "N")
    direct = model.score([combined]).scores
    np.testing.assert_allclose(paired, direct)


def test_l3_conditional_validates_sublibrary():
    with pytest.raises(ValueError, match="unknown L3 sublibrary"):
        load_model("zhou_l3_conditional", sublibrary="missing")


def test_cetnar_paper_lightgbm_bundle_loads_and_predicts_feature_rows():
    model = CetnarLightGBM.from_checkpoint_dir()
    feature_names = set()
    for member in (*model.timepoint_models.values(), model.decay_model):
        feature_names.update(member.feature_name())
    frame = pd.DataFrame(
        {name: [0] for name in feature_names if not name.startswith("predicted_")}
    )
    prediction = model.predict(frame)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_migrated_sequence_checkpoints_replay_retained_predictions():
    cases = [
        (
            "cetnar_stability",
            "AAAAAAAAAAAAAAAAAAAACCCGCCAUAUGGCGGGAGGCGGACCGAGGGAGCAGCACGACGGGAC",
            0.88651246,
        ),
        (
            "zhao_ptrc",
            "CTGACAATTAATCATCCGGCTCTTATAATGTGTGGAATTGCGGGCGGATAACAATTTCACACAGGAAACAGACC",
            3.3800247,
        ),
        (
            "zhao_ptrc_xgboost",
            "CTGACAATTAATCATCCGGCTCTTATAATGTGTGGAATTGCGGGCGGATAACAATTTCACACAGGAAACAGACC",
            3.5012786,
        ),
        (
            "tfbind8_weak",
            "ATTATGAA",
            0.11835402,
        ),
        (
            "utr5_weak",
            "CGTGAAGAGAGATTCCGCCCAGAGTGCAACCTCTACGCCAGGCGTCATCA",
            7.5493956,
        ),
        (
            "dream_weak",
            "TGCATTTTTTTCACATCTTTTTAAGTCATAACCAGTTTGGTCGGCACTGGGGCGTCCGCCTGGGCGGGAGTGCGGGAAGGTTACGGCTGTT",
            13.2746525,
        ),
    ]
    for name, sequence, expected in cases:
        prediction = load_model(name, device="cpu").score([sequence]).scores
        np.testing.assert_allclose(
            prediction.reshape(-1)[0], expected, rtol=0, atol=2e-4
        )


def test_kosuri_replay_restores_genuine_log10_units():
    model = load_model("kosuri_composability", device="cpu")
    prediction = model.score_pairs(
        ["GGCGCGCCATTAATCATCCGGAATG"],
        ["ATCTTAATCTAGCGCGGGAGTATTTCATATG"],
    ).scores
    np.testing.assert_allclose(
        prediction[0],
        [3.0738265623, -2.2585909275],
        rtol=0,
        atol=2e-4,
    )


@pytest.mark.parametrize("name", ["cetnar_stability", "zhao_ptrc", "zhao_ptrc_xgboost"])
def test_bounded_variable_length_models_honor_padding_side(name):
    sequence = "ACGT" * 10
    left = load_model(name, device="cpu", padding_side="left").score([sequence])
    right = load_model(name, device="cpu", padding_side="right").score([sequence])
    assert np.isfinite(left.scores).all()
    assert np.isfinite(right.scores).all()


def _one_hot(sequence):
    lookup = {base: index for index, base in enumerate("ACGT")}
    indices = torch.tensor([[lookup[base] for base in sequence]])
    return torch.nn.functional.one_hot(indices, num_classes=4).float()


def test_dream_legnet_soft_input_matches_discrete_preprocessing_and_gradients():
    variable = "ACGT" * 23 + "A"
    full_construct = "T" * 17 + variable
    full_model = load_model("dream_legnet", device="cpu")
    variable_model = load_model("dream_legnet", device="cpu", variable_region_only=True)

    full_discrete = full_model.score([full_construct]).scores
    variable_discrete = variable_model.score([variable]).scores
    full_probabilities = _one_hot(full_construct).requires_grad_(True)
    variable_probabilities = _one_hot(variable).requires_grad_(True)
    full_soft, _ = full_model.score_soft(full_probabilities)
    variable_soft, _ = variable_model.score_soft(variable_probabilities)

    np.testing.assert_allclose(
        full_soft.detach().numpy().reshape(-1),
        full_discrete.reshape(-1),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        variable_soft.detach().numpy().reshape(-1),
        variable_discrete.reshape(-1),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(full_discrete, variable_discrete, rtol=0, atol=1e-6)

    full_soft.sum().backward()
    variable_soft.sum().backward()
    assert torch.count_nonzero(full_probabilities.grad[:, :17]).item() == 0
    assert torch.count_nonzero(full_probabilities.grad[:, 17:]).item() > 0
    assert torch.count_nonzero(variable_probabilities.grad).item() > 0


def test_dream_legnet_runs_gradient_optimizer_on_93_variable_positions():
    model = load_model("dream_legnet", device="cpu", variable_region_only=True)
    initial = ["A" * 93, "C" * 93]
    result = fast_seqprop_optimize(
        model,
        SequenceSpace(length=93),
        initial=initial,
        num_candidates=2,
        budget=4,
        learning_rate=0.1,
        seed=7,
    )
    assert result.query_count == 4
    assert result.activities.shape == (2, 1)
    assert all(len(sequence) == 93 for sequence in result.sequences)
