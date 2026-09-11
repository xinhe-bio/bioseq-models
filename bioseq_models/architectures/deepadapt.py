"""DeepAdapt sequence-and-environment network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from entmax import entmax15


class GumbelSigmoid(nn.Module):
    """
    Apply Gumbel-Sigmoid activation.

    During inference:
        Deterministic (no Gumbel noise), but still routed through the
        straight-through estimator instead of a raw (logits > 0).float()
        step function. sigmoid(logits / tau) > 0.5 iff logits > 0, so the
        hard decision -- and therefore every deployed/inference output of
        the model -- is numerically identical to the previous
        (logits > 0).float() branch. The only change is that this gate now
        carries a well-defined backward path (d sigmoid(logits/tau) / d
        logits) instead of a gradient that is zero almost everywhere, which
        is what gradient-based attribution (DeepLIFT, Integrated Gradients,
        vanilla saliency) needs to assign credit through this gate. Pure
        forward-perturbation methods (ISM) never depended on this gradient
        and are unaffected either way.

    During training:
        Use Gumbel-Sigmoid with the straight-through estimator (unchanged).

    Args:
        logits: Input logits tensor.

    Returns:
        Tensor after Gumbel-Sigmoid sampling.
    """

    def __init__(self, tau=0.1, hard=True):
        super().__init__()
        self.tau = tau
        self.hard = hard

    def forward(self, logits):
        if self.training:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(logits) + 1e-10) + 1e-10
            )
            y = torch.sigmoid((logits + gumbel_noise) / self.tau)
        else:
            y = torch.sigmoid(logits / self.tau)

        if self.hard:
            y_hard = (y > 0.5).float()
            y = y_hard - y.detach() + y

        return y


class Linear2D(nn.Module):
    """
    Apply channel-wise linear projection to 2D or 3D input tensors.

    Args:
        A: Input tensor. Supported shapes depend on configuration.
        is_dropout: Whether to apply dropout to factor matrices.

    Returns:
        Transformed tensor after optional reduction.
    """

    def __init__(
        self,
        channel_dim,
        input_dim,
        output_dim,
        inter_dim,
        only_A=False,
        flatten_dim=None,
        input_length=None,
        add_bias=False,
        reduction=None,
        dropout_rate=None,
    ):
        super(Linear2D, self).__init__()
        self.inter_dim = inter_dim
        self.flatten_dim = flatten_dim
        self.only_A = only_A
        self.reduction = reduction
        self.dropout_rate = dropout_rate
        if flatten_dim is not None:
            self.flatten_proj = nn.Parameter(
                torch.randn(channel_dim, input_dim, flatten_dim)
            )
            self.transA = nn.Parameter(
                torch.randn(channel_dim, flatten_dim * input_length, inter_dim)
            )
            self.transB = nn.Parameter(torch.randn(channel_dim, inter_dim, output_dim))
            for param in [self.transA, self.transB, self.flatten_proj]:
                nn.init.xavier_uniform_(param)
        elif only_A:
            self.transA = nn.Parameter(torch.randn(channel_dim, input_dim, output_dim))
            nn.init.xavier_uniform_(self.transA)
        else:
            self.transA = nn.Parameter(torch.randn(channel_dim, input_dim, inter_dim))
            self.transB = nn.Parameter(torch.randn(channel_dim, inter_dim, output_dim))
            for param in [self.transA, self.transB]:
                nn.init.xavier_uniform_(param)
        self.add_bias = add_bias
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(channel_dim, output_dim))
        if dropout_rate is not None:
            self.factor_drop = nn.Dropout1d(dropout_rate)

        if reduction is None:
            self.reduction_fun = lambda x: x  # 恒等函数
        elif reduction == "mean":
            self.reduction_fun = lambda x: torch.mean(x, 0)  # 对第 0 维求均值
        elif reduction == "max":
            self.reduction_fun = lambda x: torch.max(x, 0)[0]  # 对第 0 维求均值
        elif reduction == "mlp":
            self.channel_factor = nn.Parameter(torch.randn(output_dim, channel_dim, 1))
            nn.init.xavier_uniform_(self.channel_factor)

    def forward(self, A, is_dropout=True):
        dimA = A.dim()
        if is_dropout and self.dropout_rate is not None:
            transA = self.factor_drop(self.transA)
            if self.flatten_dim is not None:
                flatten_proj = self.factor_drop(self.flatten_proj)
        else:
            transA = self.transA
            if self.flatten_dim is not None:
                flatten_proj = self.flatten_proj
        if self.flatten_dim is not None:
            output = torch.einsum("bld,cdr->cblr", A, flatten_proj)
            output = torch.flatten(output, start_dim=2)
            output = torch.einsum("cbd,cdr->cbr", output, transA)
            output = torch.einsum("cbr,cro->cbo", output, self.transB)
        elif self.only_A:
            output = torch.einsum("bd,cdr->cbr", A, transA)
        elif dimA == 3:
            output = torch.einsum("bld,cdr->cblr", A, transA)
            output = torch.einsum("cblr,cro->cblo", output, self.transB)
        elif dimA == 2:
            output = torch.einsum("bd,cdr->cbr", A, transA)
            output = torch.einsum("cbr,cro->cbo", output, self.transB)
        if self.add_bias:
            bias = self.bias
            if dimA == 2 or self.flatten_dim is not None or self.only_A:
                bias = bias.unsqueeze(1)
            else:
                bias = bias.unsqueeze(1).unsqueeze(1)
            output = output + bias
        if self.reduction == "mlp":
            output = torch.einsum("cbd,dcr->bdr", output, self.channel_factor)
            return output.squeeze(-1)
        else:
            return self.reduction_fun(output)


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 2, 1)


class Squeeze(nn.Module):
    """专门用于squeeze操作的模块"""

    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


class Gated_Attention(nn.Module):
    """
    Compute gated attention over sequence features.

    Args:
        input: Input tensor containing sequence/environment interaction features.
        with_env: Whether environmental interaction is included.

    Returns:
        Gate-weighted value tensor.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        self.input_norm = nn.Sequential(
            Transpose(), nn.ReLU(), nn.BatchNorm1d(output_dim), Transpose()
        )

        self.gate = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=input_dim, bias=False),
            Transpose(),
            nn.AvgPool1d(kernel_size=3, padding=1, stride=1),
            Transpose(),
            GumbelSigmoid(tau=0.1, hard=True),  # 严格二值门控
            # nn.Sigmoid()
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=output_dim, out_features=input_dim, bias=False),
            Transpose(),
            nn.AvgPool1d(kernel_size=3, padding=1, stride=1),
            Transpose(),
        )

    def forward(self, input, with_env=True):
        if with_env:
            input_norm = self.input_norm(input)
        else:
            input_norm = self.input_norm(input)
            input_norm = torch.concat(
                [torch.zeros_like(input_norm), input_norm],
            )
        gate = self.gate(input_norm)  # [0, 1]
        value = self.value(input_norm) / (self.output_dim**0.5)
        return gate * value


class Env_Attention(nn.Module):
    """
    Compute environment attention scores for a given species.

    Args:
        x: Input tensor of shape (batch, output_dim).
        species_id: Species index used to select the species-specific scorer.

    Returns:
        Sparse attention weights over environmental features.
    """

    def __init__(
        self, input_dim, output_dim, env_dim, bottleneck_dim=None, total_species=1
    ):
        super().__init__()
        self.output_dim = output_dim
        self.env_dim = env_dim
        self.bottleneck_dim = (
            bottleneck_dim if bottleneck_dim is not None else output_dim // 2
        )

        self.input_norm_proj = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
        )

        self.score = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(output_dim, self.bottleneck_dim, bias=False),
                    nn.Linear(self.bottleneck_dim, input_dim * env_dim, bias=False),
                )
                for _ in range(total_species)
            ]
        )

    def forward(self, x, species_id):
        norm_x = self.input_norm_proj(x)
        scores = self.score[species_id](norm_x)
        if self.env_dim > 1:
            scores = torch.stack(torch.chunk(scores, chunks=self.env_dim, dim=1), dim=2)
        else:
            scores = scores.unsqueeze(-1)
        attention = entmax15(scores, 1)
        return attention


class ResNet1D(nn.Module):
    """
    Apply a residual 1D convolution block followed by normalization, pooling, and dropout.

    Args:
        x: Input tensor of shape (batch, channels, length).

    Returns:
        Output tensor after residual addition and filtering.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        pool_size,
        pool_stride,
        pool_padding,
        dropout,
    ):
        super(ResNet1D, self).__init__()

        self.seq_encoder = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,  # 保持输入输出尺寸一致
        )

        self.seq_filter = nn.Sequential(
            Transpose(),
            nn.ReLU(),
            nn.LayerNorm(out_channels),
            Transpose(),
            nn.AvgPool1d(
                kernel_size=pool_size, stride=pool_stride, padding=pool_padding
            ),
            nn.Dropout1d(dropout),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x
        out = self.seq_encoder(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        out = self.seq_filter(out)
        return out


class SEQ_CNN(nn.Module):
    def __init__(self, p, seq_type):
        super(SEQ_CNN, self).__init__()
        self.p = p
        self.seq_type = seq_type
        self.seq_detector = nn.Sequential(
            nn.Conv1d(
                in_channels=int(p["seq_feature_dim"]),
                out_channels=int(p["filters1"]),
                kernel_size=int(p["kernel_size1"]),
                stride=int(p["filters_stride1"]),
                dilation=int(p["dilation1"]),
            ),
            Transpose(),
            nn.ReLU(),
            nn.LayerNorm(int(p["filters1"])),
            Transpose(),
            nn.AvgPool1d(
                kernel_size=int(p["pool_size1"]), stride=int(p["stride1"]), padding=1
            ),
            nn.Dropout1d(float(p["dropout1"])),
            Transpose(),
        )

        if seq_type == "cds":
            self.seq_size = int(p["max_cds_len"])
        elif seq_type == "3utr":
            self.seq_size = int(p["max_3utr_len"]) + 1
        elif seq_type == "5utr":
            self.seq_size = int(p["max_5utr_len"]) + 1

        self.output_size = self.seq_conv_shape()

    def seq_conv_shape(self):
        x_input = torch.zeros(1, int(self.p["seq_feature_dim"]), self.seq_size)
        x_output = self.seq_detector(x_input)
        flatten_shape = int(np.prod(x_output.size()))
        return flatten_shape

    def get_flatten_size(self):
        return self.seq_flatten_size

    def forward(self, seq_input):
        seq_feature = self.seq_detector(seq_input)
        return seq_feature


class Interaction_Step(nn.Module):
    def __init__(
        self,
        seq_shape,
        p,
        env_dim,
        in_channels,
        out_channels,
        env_postprocess,
        total_species=1,
        use_encoder=True,
    ):
        super(Interaction_Step, self).__init__()
        self.p = p
        self.env_dim = env_dim
        self.seq_shape = seq_shape
        self.in_channels = in_channels
        self.use_encoder = use_encoder
        self.env_postprocess = env_postprocess
        self.env_gate = Env_Attention(
            input_dim=int(p["env_feature_dim"]),
            output_dim=int(p["dense1"]),
            env_dim=env_dim,
            bottleneck_dim=int(p["bottleneck_dim"]),
            total_species=total_species,
        )
        self.self_factor = nn.Parameter(
            torch.randn(int(p["dense1"]), int(p["channel_dim"]), int(p["dense1"]))
        )
        self.env_factor = nn.Parameter(
            torch.randn(int(p["dense1"]), int(p["channel_dim"]), int(p["dense1"]))
        )
        self.attention_factor = nn.Parameter(
            torch.randn(int(p["dense1"]), int(p["channel_dim"]), int(p["dense1"]))
        )

        if use_encoder:
            use_channels = out_channels
        else:
            use_channels = in_channels

        self.seq_preprocessor = Linear2D(
            channel_dim=int(p["channel_dim"]),
            input_dim=use_channels,
            output_dim=int(p["dense1"]),
            inter_dim=int(p["inter_dim"]),
            dropout_rate=float(p["dropout2"]),
        )

        self.flatten_seq_preprocessor = Linear2D(
            channel_dim=int(p["channel_dim"]),
            input_dim=use_channels,
            input_length=int(seq_shape / in_channels),
            output_dim=int(p["dense1"]),
            inter_dim=int(p["inter_dim"]),
            dropout_rate=float(p["dropout2"]),
            flatten_dim=int(p["flatten_dim1"]),
        )

        self.seq_gate = Gated_Attention(
            input_dim=use_channels, output_dim=int(p["dense1"]) * 2
        )

        self.seq_drop = nn.Dropout1d(float(p["seq_dropout"]))

        if use_encoder:
            self.RestNet = ResNet1D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=int(p["kernel_size2"]),
                stride=int(p["filters_stride2"]),
                dilation=int(p["dilation2"]),
                pool_size=int(p["pool_size1"]),
                pool_stride=int(p["stride1"]),
                pool_padding=1,
                dropout=float(p["dropout1"]),
            )
            self.seq_encoder = nn.Sequential(Transpose(), self.RestNet, Transpose())

        self.output_size = self.seq_conv_shape()

    def seq_conv_shape(self):
        x_output = torch.zeros(
            1, int(self.seq_shape / self.in_channels), int(self.in_channels)
        )
        if self.use_encoder:
            x_output = self.seq_encoder(x_output)
        output_size = int(np.prod(x_output.size()))
        return output_size

    def forward(self, seq_features, env_array, species_id, env_encoder, with_env=True):
        """
        Perform one sequence-environment interaction step.

        Args:
            seq_features: Sequence feature tensor.
            env_array: Environmental feature tensor.
            species_id: Species IDs for species-specific environment encoders.
            with_env: Whether to include environmental interaction.

        Returns:
            seq_features: Updated sequence features.
            seq_attention: Sequence attention weights.
            env_attention: Environment attention weights.
            env_emb: Encoded environmental embeddings.
        """
        if self.use_encoder:
            seq_features = self.seq_encoder(seq_features)
        seq_emb = self.seq_preprocessor(seq_features)
        seq_flatten_emb = self.flatten_seq_preprocessor(self.seq_drop(seq_features))
        seq_self_interaction = torch.einsum(
            "cbd,cbld,dcr->blr", seq_flatten_emb, seq_emb, self.self_factor
        )
        env_attention = None
        if with_env:
            batch_size = env_array.size(0)

            unique_species = torch.unique(species_id)
            env_attention = torch.zeros(
                batch_size,
                int(self.p["env_feature_dim"]),
                self.env_dim,
                device=env_array.device,
                dtype=torch.float32,
            )
            env_emb = torch.zeros(
                batch_size,
                int(self.p["dense3"]),
                device=env_array.device,
                dtype=torch.float32,
            )
            for species_idx in unique_species:
                species_idx = species_idx.item()
                indices = torch.where(species_id.squeeze(-1) == species_idx)[0]
                env_species = env_array[indices]
                env_hidden_species = env_encoder[0][species_idx](env_species)
                seq_flatten_emb_species = seq_flatten_emb.index_select(1, indices)
                env_hidden_post = self.env_postprocess(env_hidden_species)
                env_attention_species = self.env_gate(
                    torch.einsum(
                        "cbd,dcr->br",
                        seq_flatten_emb_species * env_hidden_post,
                        self.attention_factor,
                    ),
                    species_idx,
                )
                env_attention.index_copy_(0, indices, env_attention_species.float())
                env_emb_species = env_encoder[1][species_idx](
                    env_species * env_attention_species
                )
                env_emb.index_copy_(0, indices, env_emb_species.float())

            seq_env_interaction = torch.einsum(
                "cbd,cbld,dcr->blr",
                self.env_postprocess(env_emb),
                seq_emb,
                self.env_factor,
            )
            seq_attention = self.seq_gate(
                torch.concat([seq_env_interaction, seq_self_interaction], 2)
            )
        else:
            seq_attention = self.seq_gate(seq_self_interaction, with_env=False)
        seq_features = seq_features * seq_attention
        return seq_features, seq_attention, env_attention, env_emb


class Interaction(nn.Module):
    def __init__(self, p, env_dim, seq_shape=None, total_species=1):
        super(Interaction, self).__init__()
        self.p = p

        if env_dim > 1:
            self.env_preprocess_list = nn.ModuleList(
                [
                    Linear2D(
                        channel_dim=int(p["channel_dim"]),
                        input_dim=env_dim,
                        input_length=int(p["env_feature_dim"]),
                        output_dim=int(p["dense3"]),
                        inter_dim=int(p["inter_dim"]),
                        reduction="mlp",
                        dropout_rate=float(p["env_dropout"]),
                        flatten_dim=4,
                    )
                    for _ in range(total_species)
                ]
            )
        else:
            self.env_preprocess_list = nn.ModuleList(
                [
                    nn.Sequential(
                        Squeeze(),
                        Linear2D(
                            channel_dim=int(p["env_channel_dim"]),
                            input_dim=int(p["env_feature_dim"]),
                            output_dim=int(p["dense3"]),
                            inter_dim=int(p["env_inter_dim"]),
                            reduction="mlp",
                            dropout_rate=float(p["env_dropout"]),
                        ),
                    )
                    for _ in range(total_species)
                ]
            )

        self.env_postprocess = Linear2D(
            channel_dim=int(p["channel_dim"]),
            input_dim=int(p["dense3"]),
            output_dim=int(p["dense1"]),
            inter_dim=int(p["inter_dim"]),
            dropout_rate=float(p["dropout2"]),
        )

        self.env_encoder1_list = nn.ModuleList(
            [
                nn.Sequential(
                    self.env_preprocess_list[i],
                    nn.ReLU(),
                    nn.BatchNorm1d(int(p["dense3"])),
                )
                for i in range(total_species)
            ]
        )

        self.env_encoder2_list = nn.ModuleList(
            [
                nn.Sequential(
                    self.env_preprocess_list[i],
                    nn.ReLU(),
                    nn.BatchNorm1d(int(p["dense3"])),
                )
                for i in range(total_species)
            ]
        )

        self.seq_step1 = Interaction_Step(
            seq_shape=seq_shape,
            p=p,
            env_dim=env_dim,
            in_channels=int(p["filters1"]),
            out_channels=int(p["filters1"]),
            env_postprocess=self.env_postprocess,
            total_species=total_species,
            use_encoder=False,
        )
        self.seq_step2 = Interaction_Step(
            seq_shape=self.seq_step1.output_size * 2,
            p=p,
            env_dim=env_dim,
            in_channels=int(p["filters1"]) * 2,
            out_channels=int(p["filters2"]),
            env_postprocess=self.env_postprocess,
            total_species=total_species,
        )

        self.output_size = self.seq_step2.output_size

        self.seq_flatten = nn.Flatten()

    def seq_process(self, seq_features, env_array, species_id=0):
        env_encoder = (self.env_encoder1_list, self.env_encoder2_list)
        seq_step1, seq_step1_attention, _, _ = self.seq_step1(
            seq_features=seq_features,
            env_array=env_array,
            species_id=species_id,
            env_encoder=env_encoder,
        )
        seq_step1 = torch.cat([seq_features, seq_step1_attention], 2)
        seq_step2, seq_step2_attention, env_step2_attention, env_emb = self.seq_step2(
            seq_features=seq_step1,
            env_array=env_array,
            species_id=species_id,
            env_encoder=env_encoder,
        )
        return seq_step2, seq_step2_attention, env_step2_attention, env_emb

    def forward(self, seq_features, env_array, species_id):
        """
        Forward pass for sequence-environment interaction.

        Args:
            seq_features: Encoded sequence features.
            env_array: Environmental feature tensor.
            species_id: Species IDs.
            report: Reserved flag for reporting/debugging.

        Returns:
            Final sequence features, sequence attention, environment attention, and environment embedding.
        """
        features, seq_attention, env_attention, seq_env_emb = self.seq_process(
            seq_features, env_array, species_id
        )
        return features, seq_attention, env_attention, seq_env_emb


"""
------------------------------------------
        RnaModel Class Construction
-------------------------------------------
"""


class DeepAdaptNetwork(nn.Module):
    """Three-tower CNN with transcriptome-conditioned TE prediction."""

    def __init__(self, model_config, num_celltype_species, total_species=1):
        super().__init__()
        if model_config["training"].get("env_type") != ["RNA"]:
            raise ValueError(
                "The clean model supports exactly one environment channel: RNA."
            )
        self.model_config = model_config
        hyper_use = model_config["training"]["hyper_use"]
        p = model_config[hyper_use]
        self.p = p
        self.utr5_CNN = SEQ_CNN(p=p, seq_type="5utr")
        self.utr3_CNN = SEQ_CNN(p=p, seq_type="3utr")
        self.cds_CNN = SEQ_CNN(p=p, seq_type="cds")
        env_dim = 1
        self.Interaction = Interaction(
            p=p,
            env_dim=env_dim,
            seq_shape=self.cds_CNN.output_size
            + self.utr5_CNN.output_size
            + self.utr3_CNN.output_size,
            total_species=total_species,
        )

        self.te_celltype_aggregator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        in_features=int(p["dense3"]),
                        out_features=(num_celltype_species[i]) * int(p["dense4"]),
                    ),
                    nn.Unflatten(
                        dim=-1,
                        unflattened_size=(int(p["dense4"]), num_celltype_species[i]),
                    ),
                    nn.Softmax(
                        dim=1
                    ),  # 在num_celltype_species维度归一化，每个dense1位置得到一个概率分布
                )
                for i in range(total_species)
            ]
        )

        self.te_generall_aggregator = nn.ModuleList(
            [
                nn.Sequential(
                    Linear2D(
                        channel_dim=int(p["env_channel_dim"]),
                        input_dim=int(p["dense3"]),
                        output_dim=int(p["dense4"]),
                        inter_dim=int(p["env_inter_dim"]),
                        reduction="mlp",
                    ),
                    nn.Softmax(dim=-1),
                )
                for i in range(total_species)
            ]
        )

        self.te_emb_extractor = nn.Sequential(
            nn.Dropout1d(float(p["seq_dropout"])),
            Linear2D(
                channel_dim=int(p["channel_dim"]),
                input_dim=int(p["filters2"]),
                input_length=int(self.Interaction.output_size / p["filters2"]),
                output_dim=int(p["dense2"]),
                inter_dim=int(p["inter_dim"]),
                dropout_rate=float(p["dropout2"]),
                reduction="mlp",
                flatten_dim=int(p["flatten_dim2"]),
            ),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense2"])),
            nn.Linear(in_features=int(p["dense2"]), out_features=int(p["dense4"])),
            nn.Tanh(),
        )

    def encode_sequences(self, utr5_seq, utr3_seq, cds_seq):
        """Encode each transcript once before pairing it with environments."""
        utr5_seq_features = self.utr5_CNN(F.pad(utr5_seq, pad=(0, 1)))
        utr3_seq_features = self.utr3_CNN(F.pad(utr3_seq, pad=(0, 1)))
        cds_seq_features = self.cds_CNN(cds_seq)
        return torch.concat(
            [utr5_seq_features, cds_seq_features, utr3_seq_features], dim=1
        )

    def forward_from_sequence_features(
        self, seq_features, mRNA_array, species_id, report_attention=False
    ):
        """Predict TE from precomputed sequence features and an environment."""
        te_features, seq_attention, env_attention, env_emb = self.Interaction(
            seq_features=seq_features, env_array=mRNA_array, species_id=species_id
        )
        te_env = self.te_pred(self.te_emb_extractor(te_features), env_emb, species_id)
        if report_attention:
            return te_env, seq_attention, env_attention
        return te_env

    def te_pred(self, te_features, env_emb, species_id, celltype_id=None):
        if celltype_id is not None:
            unique_species = torch.unique(species_id)
            batch_size = te_features.shape[0]
            te_predict = torch.zeros(
                batch_size, 1, device=te_features.device, dtype=torch.float32
            )
            te_selection = torch.zeros(
                batch_size, 1, device=te_features.device, dtype=torch.float32
            )
            selection_p = torch.zeros(
                batch_size,
                int(self.p["dense4"]),
                2,
                device=te_features.device,
                dtype=torch.float32,
            )
            for species_idx in unique_species:
                species_idx = species_idx.item()
                # 找出属于当前物种的所有样本
                indices = torch.where(species_id.squeeze(-1) == species_idx)[0]
                num_samples = len(indices)

                env_emb_species = env_emb[indices]  # (num_samples, dense3)
                te_features_species = te_features[indices]  # (num_samples, dense4)
                celltype_ids_species = celltype_id[indices].view(-1)  # (num_samples, 1)
                te_celltype_aggregator = self.te_celltype_aggregator[species_idx](
                    env_emb_species
                )
                te_predict_celltypes = torch.sum(
                    te_features_species.unsqueeze(-1) * te_celltype_aggregator,
                    dim=1,
                    keepdim=False,
                )  # (num_samples, num_celltype_species[species_idx])
                batch_indices_local = torch.arange(
                    num_samples, device=te_predict_celltypes.device
                )
                selected_te = te_predict_celltypes[
                    batch_indices_local, celltype_ids_species
                ].unsqueeze(1)  # (num_samples, 1)
                selected_aggregator = te_celltype_aggregator[
                    batch_indices_local, :, celltype_ids_species
                ]  # (num_samples, dense4)
                te_selection.index_copy_(0, indices, selected_te)
                te_general_aggregator = self.te_generall_aggregator[species_idx](
                    env_emb_species
                )
                te_predict_general = torch.sum(
                    te_features_species * te_general_aggregator, dim=1, keepdim=True
                )  # (num_samples, 1)
                te_predict.index_copy_(0, indices, te_predict_general)
                selection_p[indices, :, :] = torch.stack(
                    [selected_aggregator, te_general_aggregator], dim=2
                )
            return te_predict, te_selection, selection_p
        else:
            unique_species = torch.unique(species_id)
            batch_size = te_features.shape[0]
            te_predict = torch.zeros(
                batch_size, 1, device=te_features.device, dtype=torch.float32
            )
            for species_idx in unique_species:
                species_idx = species_idx.item()
                # 找出属于当前物种的所有样本
                indices = torch.where(species_id.squeeze(-1) == species_idx)[0]

                env_emb_species = env_emb[indices]
                te_features_species = te_features[indices]

                te_general_aggregator = self.te_generall_aggregator[species_idx](
                    env_emb_species
                )
                te_predict_general = torch.sum(
                    te_features_species * te_general_aggregator, dim=1, keepdim=True
                )
                te_predict.index_copy_(0, indices, te_predict_general.float())
            return te_predict

    def forward(
        self,
        utr5_seq,
        utr3_seq,
        cds_seq,
        mRNA_array,
        species_id,
        sample_mRNA_array=None,
        celltype_id=None,
        sample_celltype_id=None,
        utr5_seq_contrast=None,
        utr3_seq_contrast=None,
        cds_seq_contrast=None,
        report_attention=False,
    ):
        """
        Forward pass of the full TE prediction model.

        Modes:
            1. Inference/evaluation:
                Only the main input is used.
            2. Pretraining:
                Main sample + environment-contrast sample.
            3. Finetuning:
                Main sample + environment-contrast sample + isoform-contrast sample.

        Args:
            utr5_seq: 5'UTR input sequence tensor.
            utr3_seq: 3'UTR input sequence tensor.
            cds_seq: CDS input sequence tensor.
            mRNA_array: Main environmental feature tensor.
            species_id: Species IDs.
            sample_mRNA_array: Optional contrastive environment tensor.
            celltype_id: Optional main cell-type IDs.
            sample_celltype_id: Optional environment-contrast cell-type IDs.
            utr5_seq_contrast: Optional contrastive 5'UTR sequence tensor.
            utr3_seq_contrast: Optional contrastive 3'UTR sequence tensor.
            cds_seq_contrast: Optional contrastive CDS sequence tensor.
            report_attention: Whether to return attention maps.

        Returns:
            Different tuples depending on the current mode.
        """

        seq_features = self.encode_sequences(utr5_seq, utr3_seq, cds_seq)
        if sample_mRNA_array is not None:
            if celltype_id is None or sample_celltype_id is None:
                raise ValueError(
                    "celltype_id and sample_celltype_id are required during training"
                )
            if utr5_seq_contrast is not None:
                utr5_seq_features_contrast = self.utr5_CNN(
                    F.pad(utr5_seq_contrast, pad=(0, 1))
                )
                utr3_seq_features_contrast = self.utr3_CNN(
                    F.pad(utr3_seq_contrast, pad=(0, 1))
                )
                cds_seq_features_contrast = self.cds_CNN(cds_seq_contrast)
                seq_features_contrast = torch.concat(
                    [
                        utr5_seq_features_contrast,
                        cds_seq_features_contrast,
                        utr3_seq_features_contrast,
                    ],
                    1,
                )
                all_seq_features = torch.cat(
                    [seq_features, seq_features, seq_features_contrast], dim=0
                )
                all_env_array = torch.cat(
                    [mRNA_array, sample_mRNA_array, mRNA_array], dim=0
                )
                species_id_all = torch.cat([species_id, species_id, species_id])
                celltype_id_all = torch.cat(
                    [celltype_id, sample_celltype_id, celltype_id], dim=0
                )
                all_te_features, _, all_env_attention, env_emb = self.Interaction(
                    seq_features=all_seq_features,
                    env_array=all_env_array,
                    species_id=species_id_all,
                )
                TE_pred, TE_cell, selection_p = self.te_pred(
                    self.te_emb_extractor(all_te_features),
                    env_emb,
                    species_id_all,
                    celltype_id_all,
                )
                TE_env, TE_sample, TE_isoform = torch.chunk(TE_pred, chunks=3, dim=0)
                TE_env_cell, TE_sample_cell, TE_isoform_cell = torch.chunk(
                    TE_cell, chunks=3, dim=0
                )
                return (
                    TE_env,
                    TE_env_cell,
                    TE_sample,
                    TE_sample_cell,
                    TE_isoform,
                    TE_isoform_cell,
                    all_env_attention,
                    selection_p,
                )
            else:
                all_seq_features = torch.cat([seq_features, seq_features], dim=0)
                all_env_array = torch.cat([mRNA_array, sample_mRNA_array], dim=0)
                species_id_all = torch.cat([species_id, species_id])
                celltype_id_all = torch.cat([celltype_id, sample_celltype_id], dim=0)
                all_te_features, _, all_env_attention, env_emb = self.Interaction(
                    seq_features=all_seq_features,
                    env_array=all_env_array,
                    species_id=species_id_all,
                )
                TE_pred, TE_cell, selection_p = self.te_pred(
                    self.te_emb_extractor(all_te_features),
                    env_emb,
                    species_id_all,
                    celltype_id_all,
                )
                TE_env, TE_sample = torch.chunk(TE_pred, chunks=2, dim=0)
                TE_env_cell, TE_sample_cell = torch.chunk(TE_cell, chunks=2, dim=0)
                return (
                    TE_env,
                    TE_env_cell,
                    TE_sample,
                    TE_sample_cell,
                    all_env_attention,
                    selection_p,
                )
        else:
            return self.forward_from_sequence_features(
                seq_features, mRNA_array, species_id, report_attention=report_attention
            )
