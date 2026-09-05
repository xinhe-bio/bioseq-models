import numpy as np
import pytest

from bioseq_models import available_models, load_model, model_info


@pytest.mark.parametrize(
    ("name", "sequence", "tasks"),
    [
        ("optimus5prime", "A" * 50, 1),
        ("deepstarr", "A" * 249, 2),
        ("dream_legnet", "A" * 110, 1),
        ("zhao_ptrc", "A" * 74, 1),
        ("zhou_l3", "A" * 29, 1),
    ],
)
def test_bundled_model_loads_and_predicts(name, sequence, tasks):
    model = load_model(name, device="cpu", batch_size=2)
    prediction = model.score([sequence]).scores
    assert prediction.size == tasks
    assert np.isfinite(prediction).all()
    assert model.task_names == model_info(name).task_names


def test_registry_and_aliases():
    assert available_models() == (
        "optimus5prime",
        "deepstarr",
        "dream_legnet",
        "zhao_ptrc",
        "zhou_l3",
    )
    assert model_info("optimus").name == "optimus5prime"


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
