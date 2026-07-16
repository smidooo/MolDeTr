"""Configuration for the model. Contains all the parameters for the model, training and data."""

from dataclasses import dataclass

from typing import Optional


@dataclass(frozen=True)
class RegParamIndices:
    """Configuration for the indices of the regression parameters. Contains the indices of the regression parameters."""

    center_position_in_points: int
    line_width_in_points: int
    coupling_constant_1_in_points: int
    coupling_constant_2_in_points: int
    coupling_constant_3_in_points: int
    coupling_constant_4_in_points: int
    bounding_box_range_in_points: int

    def __len__(self):
        """Returns the number of regression parameters."""
        return len(self.__dict__)

    def __getitem__(self, item):
        """Returns the index of the regression parameter."""
        return self.__dict__[item]


@dataclass(frozen=True)
class FixedDistortions:
    """Configuration for the fixed distortions. Contains the fixed distortions for spectrum data augmentation."""

    phase_0_custom: float
    phase_1_custom: float
    base_left_custom: float
    base_right_custom: float
    custom_snr_value: float
    sigma_custom: float


@dataclass(frozen=True)
class MultClassIndices:
    """Configuration for the indices of the multiplet classes. Contains the indices of the multiplet classes."""

    _1p: int
    _2p: int
    _3p: int
    _4p: int
    _6p: int

    def __len__(self):
        """Returns the number of multiplet classes."""
        return len(self.__dict__)


@dataclass(frozen=True)
class Paths:
    """Configuration for the paths. Contains the path to the data folder and the path to the model folder."""

    data_folder: str
    model_folder_save: str
    experiment_folder: str
    experimental_file: str
    specific_evaluation_set: str


@dataclass(frozen=True)
class LogNames:
    """Configuration for the log names. Contains the name of the log file and the name of the log folder."""

    best_model_file: str
    extrema_file: str


@dataclass(frozen=True)
class LossWeighting:
    """Configuration for the loss weighting. Contains the weighting for the indivdual loss functions classification loss weighting equals 1, parameter_loss_weighting and giou lossi_weighting are the weighting for the respective individual loss functions."""

    classification_loss_weighting: float
    parameter_loss_weighting: float
    giou_loss_weighting: float
    parameter_weighting: Optional[list[float]]
    focal_loss_gamma: float
    focal_loss_alpha: float


@dataclass(frozen=True)
class CostWeighting:
    """Configuration for the cost weighting. Contains the weighting for the cost function."""

    parameter_cost_weighting: float
    giou_cost_weighting: float
    parameter_weighting: Optional[list[float]]


@dataclass(frozen=True)
class Weighting:
    """Configuration for the weighting. Contains the weighting for the loss function and the weighting for the parameters."""

    loss_weighting: LossWeighting
    cost_weighting: CostWeighting


@dataclass(frozen=True)
class Files:
    """Configuration for the data. Contains the name of the data file and the number of multiplets to predict. As well as the maximum number of multiplets in the data."""

    data_filename: str
    num_multiplet_pred: int


@dataclass(frozen=True)
class Device:
    """Configuration for the device. Contains the device name to use and train the model on.

    When device_name is 'cuda' but CUDA is not available, automatically falls back to 'cpu'.
    """

    device_name: str

    def __post_init__(self) -> None:
        if self.device_name.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                object.__setattr__(self, "device_name", "cpu")


def resolve_device(cfg_device_name: str) -> str:
    """Return cfg device name, falling back to 'cpu' when CUDA unavailable."""
    import torch

    if cfg_device_name.startswith("cuda") and not torch.cuda.is_available():
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():  # Apple Silicon
            return "mps"
        return "cpu"
    return cfg_device_name


@dataclass(frozen=True)
class Backbone:
    """Configuration for the backbone. Contains the input length, kernel size, number of layers in the pyramid, number of channels in the upsampling, pool size and the number of output channels of the CNN."""

    input_length: int
    kernel_size: int
    pyramid_layers: int
    channel_dim_up: int
    pool_size: int


@dataclass(frozen=True)
class Head:
    """Configuration for the head. Contains the width of the dnn and the number of hidden layers."""

    dnn_width: int
    dnn_hidden: int


@dataclass(frozen=True)
class DeformableTransformer:
    deformable_flag: bool
    n_levels: int
    n_points: int
    channel_size: int
    resnet: bool


from dataclasses import dataclass, field


@dataclass(frozen=True)
class Transformer:
    """Configuration for the transformer. Contains the number of features, number of heads, number of encoder and decoder layers, dimension of the feedforward layer and the dropout rate."""

    positional_encoding: str = field(metadata={"allowed": ["sine", "learned"]})
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    deformable: DeformableTransformer

    def __post_init__(self):
        allowed_values = self.__dataclass_fields__["positional_encoding"].metadata.get(
            "allowed", []
        )
        if self.positional_encoding not in allowed_values:
            raise ValueError(
                f"Invalid positional_encoding '{self.positional_encoding}'. Allowed values are {allowed_values}."
            )


@dataclass(frozen=True)
class ModelParams:
    """Configuration for the model. Contains the input size, number of parameters, number of classes, kernel size, width of the dnn, number of hidden layers and the number of layers in the pyramid."""

    backbone: Backbone
    head: Head
    transformer: Transformer


@dataclass(frozen=True)
class LearningRates:
    """Configuration for the learning rates. Contains the learning rate for the CNN and the transformer."""

    lr: float
    lr_backbone: float
    lr_linear_proj_mult: float


@dataclass(frozen=True)
class OptimParams:
    """Configuration for the optimizer. Contains the learning rate, weight decay, number of epochs, batch size and number of workers. As well as the number of epochs after which the learning rate is reduced and the number of epochs after which the training is stopped if the validation loss does not improve."""

    learning_rates: LearningRates
    weight_decay: float
    n_epochs_max: int
    batch_size: int
    samples_per_epoch: int
    num_workers: int
    lr_reduce_epochs: float
    early_stopping_epochs: float
    n_groups: int


@dataclass(frozen=True)
class Pretrained:
    """Configuration for the pretrained model. Decide whether to use a pretrained model or not. Folder where the model is stored."""

    use_pretrained: bool
    model_folder_load: str


@dataclass(frozen=True)
class DataAugmentation:
    """Configuration for the data augmentation. Decide whether to use a data augmentation or not. Folder where the meta file is stored."""

    flag: bool
    meta_file_name: str
    use_custom_values: bool


@dataclass(frozen=True)
class Plotting:
    base_freq_in_MHz: 400.0
    max_ppm: 7.0
    min_ppm: 6.0


@dataclass(frozen=True)
class SingleMultipletConfig:
    """Configuration for the single multiplet model. Contains all the parameters for the model, training and data."""

    reg_param_indices: RegParamIndices
    mult_class_indices: MultClassIndices
    paths: Paths
    log_names: LogNames
    files: Files
    weighting: Weighting
    pretrained: Pretrained
    data_augmentation: DataAugmentation
    model_params: ModelParams
    optim_params: OptimParams
    device: Device


@dataclass(frozen=True)
class MultipletConfig:
    """Configuration for the multiplet model. Contains all the parameters for the model, training and data."""

    reg_param_indices: RegParamIndices
    mult_class_indices: MultClassIndices
    paths: Paths
    lognames: LogNames
    files: Files
    fixed_distortions: FixedDistortions
    weighting: Weighting
    pretrained: Pretrained
    data_augmentation: DataAugmentation
    model_params: ModelParams
    optim_params: OptimParams
    device: Device
    plotting: Plotting
