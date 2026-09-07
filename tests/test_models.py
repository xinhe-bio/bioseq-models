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
    assert model_info("dream_legnet").task_names == ("promoter_activity",)
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
def test_fixed_model_supports_shorter_padded_sequences(padding_side):
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


def test_cetnar_returns_measured_decay_rate():
    model = load_model("cetnar_stability", device="cpu", padding_side="left")
    prediction = model.score(["ACGT" * 10, "ACGT" * 15])
    assert prediction.shape == (2, 1)
    assert model.task_names == ("decay_rate_k_fit",)


def test_cetnar_checkpoint_replays_retained_prediction():
    sequence = (
        "AAAAAAAAAAAAAAAAAAAACCCGCCAUAUGGCGGGAGGCGGACCGAGGGAGCAGCACGACGGGAC"
    )
    prediction = load_model("cetnar_stability", device="cpu").score([sequence])
    np.testing.assert_allclose(prediction[0, 0], 0.88651246, rtol=0, atol=2e-4)


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


def test_invalid_nucleotide_raises_value_error():
    model = load_model("optimus5prime", device="cpu")
    with pytest.raises(ValueError, match="invalid nucleotide"):
        model.score(["A" * 49 + "X"])
