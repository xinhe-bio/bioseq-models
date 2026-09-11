import numpy as np
import pytest

from bioseq_models import (
    available_models,
    download_model,
    load_model,
    model_info,
    resources,
)

EMBRYO_ACCESSIBILITY_TISSUES = (
    "brain",
    "cns",
    "epidermis",
    "midgut",
    "somatic_muscle",
    "visceral_muscle",
)
EMBRYO_ACTIVITY_TISSUES = (
    "brain_vnc",
    "muscle",
    "epidermis",
    "gut",
    "brain_specific",
)

MODEL_CASES = [
    ("optimus5prime", "A" * 50, 1, {}),
    ("deepstarr", "A" * 249, 2, {}),
    ("deepstarr_mouse_heart_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_heart_accessibility", "A" * 1001, 1, {}),
    ("deepstarr_mouse_limb_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_limb_accessibility", "A" * 1001, 1, {}),
    ("deepstarr_mouse_midbrain_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_midbrain_accessibility", "A" * 1001, 1, {}),
    ("deepstarr_mouse_forebrain_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_forebrain_accessibility", "A" * 1001, 1, {}),
    ("deepstarr_mouse_hindbrain_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_hindbrain_accessibility", "A" * 1001, 1, {}),
    ("deepstarr_mouse_neural_tube_activity", "A" * 1001, 1, {}),
    ("deepstarr_mouse_neural_tube_accessibility", "A" * 1001, 1, {}),
    *(
        (f"deepstarr_embryo_{tissue}_accessibility", "A" * 1001, 1, {})
        for tissue in EMBRYO_ACCESSIBILITY_TISSUES
    ),
    *(
        (f"deepstarr_embryo_{tissue}_activity", "A" * 1001, 1, {})
        for tissue in EMBRYO_ACTIVITY_TISSUES
    ),
    ("malinois", "A" * 200, 3, {}),
    ("dream_legnet", "A" * 110, 1, {}),
    ("cetnar_stability", "A" * 60, 1, {}),
    (
        "utr_code",
        "ACGT" * 25,
        1,
        {
            "design_region": "utr3",
            "utr5": "ACGT" * 20,
            "cds": "ATG" + "GCT" * 18 + "TAA",
        },
    ),
    (
        "deepadapt_heatshock",
        "GCGT" * 20,
        3,
        {
            "design_region": "utr5",
            "cds": "ATG" + "GCT" * 30 + "TAA",
            "utr3": "ACGT" * 20,
        },
    ),
    (
        "cds_stability",
        "ATG" + "GCT" * 18 + "TAA",
        1,
        {"species": "human"},
    ),
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
    assert model_info("deepstarr_mouse").name == "deepstarr_mouse_heart_activity"
    assert model_info("deepstarr_mouse_cns").name == "deepstarr_mouse_midbrain_activity"
    assert (
        model_info("deepstarr_mouse_cns_accessibility").name
        == "deepstarr_mouse_midbrain_accessibility"
    )
    assert model_info("deepstarr_embryo").name == "deepstarr_embryo_brain_vnc_activity"
    assert model_info("dream_legnet").task_names == ("promoter_activity",)
    assert model_info("utrcode").name == "utr_code"
    assert model_info("heatshock").name == "deepadapt_heatshock"
    assert model_info("mrna_stability").name == "cds_stability"
    assert set(available_models(status="paper_architecture")) == {
        "optimus5prime",
        "deepstarr",
        "dream_legnet",
        "parm_k562",
        "agarwal_hepg2",
        "agarwal_k562",
        "agarwal_wtc11",
    }
    assert available_models(status="official_checkpoint") == ("malinois",)
    expected_converted = {
        "deepstarr_mouse_heart_activity",
        "deepstarr_mouse_heart_accessibility",
        "deepstarr_mouse_limb_activity",
        "deepstarr_mouse_limb_accessibility",
        "deepstarr_mouse_midbrain_activity",
        "deepstarr_mouse_midbrain_accessibility",
        "deepstarr_mouse_forebrain_activity",
        "deepstarr_mouse_forebrain_accessibility",
        "deepstarr_mouse_hindbrain_activity",
        "deepstarr_mouse_hindbrain_accessibility",
        "deepstarr_mouse_neural_tube_activity",
        "deepstarr_mouse_neural_tube_accessibility",
        *(
            f"deepstarr_embryo_{tissue}_accessibility"
            for tissue in EMBRYO_ACCESSIBILITY_TISSUES
        ),
        *(
            f"deepstarr_embryo_{tissue}_activity"
            for tissue in EMBRYO_ACTIVITY_TISSUES
        ),
    }
    assert set(available_models(status="converted_official_checkpoint")) == (
        expected_converted
    )


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
    sequence = "AAAAAAAAAAAAAAAAAAAACCCGCCAUAUGGCGGGAGGCGGACCGAGGGAGCAGCACGACGGGAC"
    prediction = load_model("cetnar_stability", device="cpu").score([sequence])
    np.testing.assert_allclose(prediction[0, 0], 0.88651246, rtol=0, atol=2e-4)


def test_malinois_official_checkpoint_replays_retained_prediction():
    prediction = load_model("malinois", device="cpu").score(["ACGT" * 50])
    np.testing.assert_allclose(
        prediction[0],
        [9.92313004, 8.74214935, 16.36242104],
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("deepstarr_mouse_heart_activity", 0.0507650263607502),
        ("deepstarr_mouse_heart_accessibility", 0.787926197052002),
        ("deepstarr_mouse_limb_activity", 0.17488349974155426),
        ("deepstarr_mouse_limb_accessibility", 1.4313772916793823),
        ("deepstarr_mouse_midbrain_activity", 0.3002518117427826),
        ("deepstarr_mouse_midbrain_accessibility", 3.071046829223633),
        ("deepstarr_mouse_forebrain_activity", 0.22948212921619415),
        ("deepstarr_mouse_forebrain_accessibility", 2.5464158058166504),
        ("deepstarr_mouse_hindbrain_activity", 0.21791881322860718),
        ("deepstarr_mouse_hindbrain_accessibility", 2.260223150253296),
        ("deepstarr_mouse_neural_tube_activity", 0.14803923666477203),
        ("deepstarr_mouse_neural_tube_accessibility", 2.2527151107788086),
        ("deepstarr_embryo_brain_vnc_activity", 0.17912831902503967),
        ("deepstarr_embryo_brain_accessibility", -9.825883865356445),
    ],
)
def test_tissue_deepstarr_checkpoints_replay_retained_predictions(name, expected):
    prediction = load_model(name, device="cpu").score(["A" * 1001])
    np.testing.assert_allclose(prediction[0, 0], expected, rtol=0, atol=2e-6)


def test_deepstarr_mouse_uses_all_official_folds_and_replicates():
    model = load_model("deepstarr_mouse_limb_activity", device="cpu")
    assert model.ensemble_size == 6


def test_deepstarr_embryo_uses_all_usable_official_folds_and_replicates():
    model = load_model("deepstarr_embryo_brain_vnc_activity", device="cpu")
    assert model.ensemble_size == 22


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


def test_utr_code_replays_retained_prediction():
    model = load_model(
        "utr_code",
        device="cpu",
        batch_size=1,
        design_region="utr3",
        utr5="ACGT" * 20,
        cds="ATG" + "GCT" * 98 + "TAA",
    )
    prediction = model.score(["ACGT" * 25])
    np.testing.assert_allclose(prediction[0, 0], 7.63377285, rtol=0, atol=1e-6)


def test_deepadapt_heatshock_replays_retained_prediction():
    model = load_model(
        "deepadapt_heatshock",
        device="cpu",
        batch_size=1,
        design_region="utr5",
        cds="ATG" + "GCT" * 30 + "TAA",
        utr3="ACGT" * 20,
    )
    prediction = model.score(["GCGT" * 20])
    np.testing.assert_allclose(
        prediction[0],
        [-0.47076094, 1.86393166, 1.39317071],
        rtol=0,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        prediction[:, 0], prediction[:, 2] - prediction[:, 1], rtol=0, atol=1e-7
    )


def test_cds_stability_validates_complete_codon_input():
    model = load_model("cds_stability", device="cpu", species="human")
    with pytest.raises(ValueError, match="divisible by 3"):
        model.score(["ATGA"])
    with pytest.raises(ValueError, match="contains N"):
        model.score(["ATGNNNTAA"])


def test_hugging_face_download_is_scoped_to_one_model(monkeypatch, tmp_path):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return tmp_path / kwargs["filename"]

    monkeypatch.setattr(resources, "hf_hub_download", fake_download)
    downloaded = download_model("utr_code", cache_dir=tmp_path)

    assert len(downloaded) == 2
    assert [call["filename"] for call in calls] == [
        "utr_code/best.epoch",
        "utr_code/HEK293T.txt",
    ]
    assert all(call["repo_id"] == "xinHesysu/bioseq-models" for call in calls)
    assert all(call["revision"] == "v0.10.0" for call in calls)


def test_deepstarr_mouse_download_fetches_one_six_checkpoint_ensemble(
    monkeypatch, tmp_path
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return tmp_path / kwargs["filename"]

    monkeypatch.setattr(resources, "hf_hub_download", fake_download)
    downloaded = download_model("deepstarr_mouse_cns", cache_dir=tmp_path)

    assert len(downloaded) == 6
    assert [call["filename"] for call in calls] == [
        f"checkpoints/deepstarr_mouse_midbrain_activity_fold{fold:02d}_rep{rep}.pt"
        for fold in range(1, 4)
        for rep in range(1, 3)
    ]


def test_deepstarr_embryo_download_fetches_one_checkpoint_ensemble(
    monkeypatch, tmp_path
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return tmp_path / kwargs["filename"]

    monkeypatch.setattr(resources, "hf_hub_download", fake_download)
    downloaded = download_model(
        "deepstarr_embryo_visceral_muscle_accessibility",
        cache_dir=tmp_path,
    )

    assert len(downloaded) == 29
    assert [call["filename"] for call in calls] == [
        "checkpoints/deepstarr_embryo_visceral_muscle_accessibility_"
        f"fold{fold:02d}_rep{rep}.pt"
        for fold in range(1, 11)
        for rep in range(1, 4)
        if (fold, rep) != (9, 3)
    ]
