import numpy as np
import torch
from torch import nn

from bioseq_models.adapters.common import (
    encode_acgt_sequences,
    prepare_acgt_probabilities,
)
from bioseq_models.inference import TorchSequenceModel


class _WeightedChannels(nn.Module):
    def forward(self, values):
        weights = torch.as_tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=values.dtype, device=values.device
        )
        return (values * weights[None, :, None]).sum(dim=(1, 2), keepdim=False)[:, None]


def test_torch_sequence_model_discrete_and_soft_paths_agree_and_preserve_gradients():
    scorer = TorchSequenceModel(
        _WeightedChannels(),
        lambda values: encode_acgt_sequences(values, exact_length=4),
        soft_encoder=lambda values: prepare_acgt_probabilities(
            values, exact_length=4
        ),
        task_names=("activity",),
    )
    sequences = np.asarray(["ACGT", "GGGG"])
    probabilities = torch.nn.functional.one_hot(
        torch.as_tensor([[0, 1, 2, 3], [2, 2, 2, 2]]), num_classes=4
    ).float().requires_grad_()

    soft, uncertainty = scorer.score_soft(probabilities)
    np.testing.assert_allclose(soft.detach().numpy(), scorer.score(sequences))
    assert uncertainty is None
    soft.sum().backward()
    assert probabilities.grad is not None
    assert scorer.supports_soft_scoring


def test_noncanonical_model_channel_order_is_owned_by_soft_encoder():
    canonical = torch.eye(4).unsqueeze(0)
    agct = prepare_acgt_probabilities(
        canonical,
        exact_length=4,
        channel_order="AGCT",
    )
    assert agct.argmax(dim=1).tolist() == [[0, 2, 1, 3]]
