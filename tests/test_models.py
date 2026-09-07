import numpy as np
import pytest

from bioseq_models import available_models, load_model, model_info


MODEL_CASES = [
    ("optimus5prime", "A" * 50, 1, {}),
    ("deepstarr", "A" * 249, 2, {}),
    ("dream_legnet", "A" * 110, 1, {}),
    ("cetnar_stability", "A" * 60, 1, {}),
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
def test_bundled_model_loads_and_returns_numpy(name, sequence, tasks, options):
    model = load_model(name, device="cpu", batch_size=2, **options)
    prediction = model.score([sequence])
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, tasks)
    assert np.isfinite(prediction).all()
    assert model.task_names == model_info(name).task_names


def test_registry_and_aliases():
    assert available_models() == tuple(case[0] for case in MODEL_CASES)
    assert model_info("optimus").name == "optimus5prime"
    assert model_info("dream").sequence_length == 110
    assert model_info("tfbind8").status == "weak_surrogate"
    assert tuple(model_info("dream_legnet").task_names) == ("promoter_activity",)
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
    prediction = model.score(["ACGU" * 10])
    assert prediction.shape == (1, 1)
    assert np.isfinite(prediction).all()


def test_rna_u_is_equivalent_to_dna_t():
    model = load_model("optimus5prime", device="cpu")
    rna = model.score(["ACGU" * 12 + "AC"])
    dna = model.score(["ACGT" * 12 + "AC"])
    np.testing.assert_allclose(rna, dna)


def test_cetnar_returns_only_measured_decay_rate():
    model = load_model("cetnar_stability", device="cpu", padding_side="left")
    prediction = model.score(["ACGT" * 10, "ACGT" * 15])
    assert prediction.shape == (2, 1)
    assert model.task_names == ("decay_rate_k_fit",)


def test_kosuri_pair_api_matches_canonical_combined_slots():
    model = load_model("kosuri_composability", device="cpu")
    promoter, rbs = "ACGT" * 10, "TGCA" * 5
    paired = model.score_pairs([promoter], [rbs])
    combined = promoter.ljust(75, "N") + rbs.ljust(34, "N")
    direct = model.score([combined])
    np.testing.assert_allclose(paired, direct)


def test_l3_conditional_validates_sublibrary():
    with pytest.raises(ValueError, match="unknown L3 sublibrary"):
        load_model("zhou_l3_conditional", sublibrary="missing")


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
        ("tfbind8_weak", "ATTATGAA", 0.11835402),
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
        prediction = load_model(name, device="cpu").score([sequence])
        np.testing.assert_allclose(prediction[0, 0], expected, rtol=0, atol=2e-4)


def test_kosuri_replay_restores_genuine_log10_units():
    model = load_model("kosuri_composability", device="cpu")
    prediction = model.score_pairs(
        ["GGCGCGCCATTAATCATCCGGAATG"],
        ["ATCTTAATCTAGCGCGGGAGTATTTCATATG"],
    )
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
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()


def test_dream_legnet_full_and_variable_region_inputs_match():
    variable = "ACGT" * 23 + "A"
    full_construct = "T" * 17 + variable
    full_model = load_model("dream_legnet", device="cpu")
    variable_model = load_model("dream_legnet", device="cpu", variable_region_only=True)
    np.testing.assert_allclose(
        full_model.score([full_construct]),
        variable_model.score([variable]),
        rtol=0,
        atol=1e-6,
    )


def test_empty_batch_has_stable_two_dimensional_shape():
    prediction = load_model("deepstarr", device="cpu").score([])
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (0, 2)
