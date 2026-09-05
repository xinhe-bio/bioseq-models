from bioseq_models import load_model
from sequence_optimization import SequenceSpace, genetic_algorithm_optimize


def test_model_can_be_passed_directly_to_optimizer():
    model = load_model(
        "optimus5prime",
        device="cpu",
        pad_to_length=50,
        padding_side="right",
    )
    initial = ["A" * 40, "C" * 40, "G" * 40, "T" * 40]
    result = genetic_algorithm_optimize(
        model,
        SequenceSpace(length=40),
        initial=initial,
        num_candidates=4,
        budget=8,
        mutation_rate=0.1,
        seed=4,
    )
    assert result.activities.shape == (4, 1)
    assert all(len(sequence) == 40 for sequence in result.sequences)
