"""This module contains the function to initialize a fastai learner object for the multi multiplet model."""

import fastai
import torch.multiprocessing
from fastai.vision.all import *

from moldetr.dataloader.data_augmentation import augment_distortions
from moldetr.model.deformable_detr_nmr import Deformable_DETR_NMR
from moldetr.model.deformable_transformer import DeformableTransformer
from moldetr.model.positional_embedding import LearnedPositionalEncoding

from moldetr.model.utils import ParamEmbedding, get_param_groups

import sys as _sys
if _sys.platform != "win32":
    # "file_system" sharing is a Linux/macOS strategy; skip on Windows (spawn start-method).
    torch.multiprocessing.set_sharing_strategy("file_system")

from moldetr.config import MultipletConfig
from moldetr.dataloader.dataloader import get_train_test_set
from moldetr.dataloader.normalization import NormalizationParams
from moldetr.dataloader.transforms import Normalize
from moldetr.loss.combined_loss import (
    combined_loss_func,
)
from moldetr.loss.individual_losses import (
    parameter_loss,
    classification_loss,
    giou_loss,
    calculate_giou,
    single_parameter_loss,
    sigmoid_focal_loss,
)
from moldetr.matcher.matcher import matching
from moldetr.metrics.multiplet_metrics import (
    accuracy_with_empty_object,
    accuracy_without_empty_object,
    parameter_loss_metric,
    giou_loss_metric,
    classification_loss_metric,
    single_parameter_loss_metric,
)

from moldetr.model.fpn_backbone import FPN_BB


def init_learner(cfg: MultipletConfig, test: bool = False) -> fastai.learner.Learner:
    """Initialize a fastai learner object for the multi multiplet model. The learner object is used to train the model. The function returns the learner object. The function also saves the learner object in the log folder.
    Args:
        cfg: configuration object
        test: if True, the learner object is initialized for testing
    Returns: fastai learner object
    """

    # concat path
    root_path = Path(__file__).parents[2]
    data_path = root_path / cfg.paths.data_folder / cfg.files.data_filename

    # get extrema dict
    extrema = NormalizationParams(
        data_path,
        cfg.lognames.extrema_file,
        cfg.model_params.backbone.input_length,
        cfg.reg_param_indices,
    ).get_extrema()

    # data augmentation
    if (
        cfg.data_augmentation.flag
        and test
        and cfg.data_augmentation.use_custom_values
    ):
        with open(
            data_path / (cfg.data_augmentation.meta_file_name + ".json"), "r"
        ) as f:
            data_augmentation_meta = json.load(f)

        data_augmentation_partial = partial(
            augment_distortions,
            **cfg.fixed_distortions,
            ppm_right=data_augmentation_meta["ppm_right"],
            ppm_left=data_augmentation_meta["ppm_left"],
            use_custom_values=cfg.data_augmentation.use_custom_values,
        )
    elif cfg.data_augmentation.flag:
        with open(
            data_path / (cfg.data_augmentation.meta_file_name + ".json"), "r"
        ) as f:
            data_augmentation_meta = json.load(f)

        data_augmentation_partial = partial(
            augment_distortions,
            ppm_right=data_augmentation_meta["ppm_right"],
            ppm_left=data_augmentation_meta["ppm_left"],
            use_custom_values=cfg.data_augmentation.use_custom_values,
        )
    else:
        data_augmentation_partial = None
    # Define the number of query groups and the size of each query group
    batch_size = cfg.optim_params.batch_size
    # dataloader
    if test:
        _, dls = get_train_test_set(
            data_dir=data_path,
            batch_size=batch_size,
            num_classes=len(cfg.mult_class_indices),
            reg_param_indices=cfg.reg_param_indices,
            num_workers=cfg.optim_params.num_workers,
            transformation=Normalize(extrema),
            samples_per_epoch=cfg.optim_params.samples_per_epoch,
            data_augmentation=data_augmentation_partial,
            specific_evaluation_set=cfg.paths.specific_evaluation_set,
            test=True,
        )

    else:
        dls, _ = get_train_test_set(
            data_dir=data_path,
            batch_size=batch_size,
            num_classes=len(cfg.mult_class_indices),
            reg_param_indices=cfg.reg_param_indices,
            num_workers=cfg.optim_params.num_workers,
            transformation=Normalize(extrema),
            samples_per_epoch=cfg.optim_params.samples_per_epoch,
            data_augmentation=data_augmentation_partial,
        )

    # dataloader to device
    from moldetr.config import resolve_device

    device = resolve_device(cfg.device.device_name)
    dls.to(device)

    # model: FPN backbone + learned positional encoding + Deformable DETR head.
    backbone = FPN_BB(
        input_length=cfg.model_params.backbone.input_length,
        number_of_classes=len(cfg.mult_class_indices),
        num_multiplet_pred=cfg.files.num_multiplet_pred // cfg.optim_params.n_groups,
        kernel_size=cfg.model_params.backbone.kernel_size,
        num_params=len(cfg.reg_param_indices),
        pyramid_layers=cfg.model_params.backbone.pyramid_layers,
        channel_dim_up=cfg.model_params.backbone.channel_dim_up,
        pool_size=cfg.model_params.backbone.pool_size,
        cnn_output_dimension=cfg.model_params.backbone.channel_dim_up,
    ).to(device)

    positional_encoding = LearnedPositionalEncoding(
        d_model=cfg.model_params.transformer.d_model,
        max_len=cfg.model_params.backbone.input_length,
    ).to(device)

    parameter_embedding = ParamEmbedding(
        num_params=len(cfg.reg_param_indices),
        hidden_dim=cfg.model_params.transformer.d_model,
        num_decoder_layers=cfg.model_params.transformer.num_decoder_layers,
    ).to(device)

    transformer = DeformableTransformer(
        d_model=cfg.model_params.transformer.d_model,
        nhead=cfg.model_params.transformer.nhead,
        num_encoder_layers=cfg.model_params.transformer.num_encoder_layers,
        num_decoder_layers=cfg.model_params.transformer.num_decoder_layers,
        dim_feedforward=cfg.model_params.transformer.dim_feedforward,
        dropout_ratio=cfg.model_params.transformer.dropout,
        n_levels=cfg.model_params.transformer.deformable.n_levels,
        n_points=cfg.model_params.transformer.deformable.n_points,
        param_embed=parameter_embedding.parameter_embed,
    ).to(device)

    full_model = Deformable_DETR_NMR(
        backbone=backbone,
        positional_encoding=positional_encoding,
        transformer=transformer,
        num_classes=len(cfg.mult_class_indices),
        num_params=len(cfg.reg_param_indices),
        num_queries=cfg.files.num_multiplet_pred,
        hidden_dim=cfg.model_params.transformer.d_model,
        backbone_output_dim=cfg.model_params.backbone.channel_dim_up,
        n_groups=cfg.optim_params.n_groups,
        d_model=cfg.model_params.transformer.d_model,
        n_levels=cfg.model_params.transformer.deformable.n_levels,
        channel_size=cfg.model_params.transformer.deformable.channel_size,
        parameter_embed=parameter_embedding.parameter_embed,
    ).to(device)

    # optimizer function
    # opt_func = partial(
    #     OptimWrapper,
    #     opt=torch.optim.RAdam,
    #     # opt=torch.optim.AdamW,
    #     lr=cfg.optim_params.learning_rate,
    #     weight_decay=cfg.optim_params.weight_decay,
    # )

    opt_func = partial(
        RAdam,
        # params=params_dict,
        lr=cfg.optim_params.learning_rates.lr,
        wd=cfg.optim_params.weight_decay,
    )

    # loss  weighting for parameters
    if cfg.weighting.loss_weighting.parameter_weighting is None:
        parameter_loss_weights = None
    else:
        parameter_loss_weights = torch.tensor(
            cfg.weighting.loss_weighting.parameter_weighting,
            device=device,
        )
    # cost weighting for parameters
    if cfg.weighting.cost_weighting.parameter_weighting is None:
        parameter_cost_weights = None
    else:
        parameter_cost_weights = torch.tensor(
            cfg.weighting.cost_weighting.parameter_weighting,
            device=device,
        )
    # GIoU calculation for loss, cost and metrics
    calculate_giou_partial = partial(
        calculate_giou,
        parameter_indices=cfg.reg_param_indices,
        transform=Normalize(extrema),
    )
    # Matcher
    matcher_partial = partial(
        matching,
        calculate_giou=calculate_giou_partial,
        n_classes=len(cfg.mult_class_indices),
        cost_weighting=cfg.weighting.cost_weighting,
        parameter_cost_weights=parameter_cost_weights,
    )
    # Individual Losses: Parameter Loss, Classification Loss, GIoU Loss

    giou_loss_partial = partial(
        giou_loss,
        calculate_giou=calculate_giou_partial,
        n_classes=len(cfg.mult_class_indices),
    )

    parameter_loss_partial = partial(
        parameter_loss,
        n_classes=len(cfg.mult_class_indices),
        parameter_weighting=parameter_loss_weights,
    )
    # TODO: Just temporary for debugging
    single_parameter_loss_partial = partial(
        single_parameter_loss,
        n_classes=len(cfg.mult_class_indices),
        parameter_weighting=parameter_loss_weights,
    )

    sigmoid_focal_loss_partial = partial(
        sigmoid_focal_loss,
        alpha=cfg.weighting.loss_weighting.focal_loss_alpha,
        gamma=cfg.weighting.loss_weighting.focal_loss_gamma,
    )

    classification_loss_partial = partial(
        classification_loss,
        loss_function=sigmoid_focal_loss_partial,
        n_classes=len(cfg.mult_class_indices),
    )

    # Combined Loss
    combined_loss = partial(
        combined_loss_func,
        parameter_loss_partial=parameter_loss_partial,
        classification_loss_partial=classification_loss_partial,
        giou_loss_partial=giou_loss_partial,
        matching_partial=matcher_partial,
        loss_weighting=cfg.weighting.loss_weighting,
        n_groups=cfg.optim_params.n_groups,
    )
    #  Metrics
    accuracy_with_empty_object_ = partial(
        accuracy_with_empty_object,
        matching_partial=matcher_partial,
        number_of_classes=len(cfg.mult_class_indices),
        n_groups=cfg.optim_params.n_groups,
    )
    accuracy_without_empty_object_ = partial(
        accuracy_without_empty_object,
        matching_partial=matcher_partial,
        number_of_classes=len(cfg.mult_class_indices),
        n_groups=cfg.optim_params.n_groups,
    )
    giou_metric = partial(
        giou_loss_metric,
        matching_partial=matcher_partial,
        giou_loss_callable=giou_loss_partial,
        giou_loss_weighting=cfg.weighting.loss_weighting.giou_loss_weighting,
        n_groups=cfg.optim_params.n_groups,
    )
    parameter_metric = partial(
        parameter_loss_metric,
        matching_partial=matcher_partial,
        parameter_loss_callable=parameter_loss_partial,
        parameter_loss_weighting=cfg.weighting.loss_weighting.parameter_loss_weighting,
        n_groups=cfg.optim_params.n_groups,
    )
    single_parameter_metric = partial(
        single_parameter_loss_metric,
        matching_partial=matcher_partial,
        single_parameter_loss_callable=single_parameter_loss_partial,
        parameter_loss_weighting=cfg.weighting.loss_weighting.parameter_loss_weighting,
        n_groups=cfg.optim_params.n_groups,
    )
    classification_metric = partial(
        classification_loss_metric,
        matching_partial=matcher_partial,
        classification_loss_callable=classification_loss_partial,
        classification_loss_weighting=cfg.weighting.loss_weighting.classification_loss_weighting,
        n_groups=cfg.optim_params.n_groups,
    )

    metrics = [
        classification_metric,
        parameter_metric,
        single_parameter_metric,
        giou_metric,
        accuracy_with_empty_object_,
        accuracy_without_empty_object_,
    ]

    # callbacks
    callbacks = [
        ReduceLROnPlateau(patience=cfg.optim_params.lr_reduce_epochs, factor=2),
        EarlyStoppingCallback(patience=cfg.optim_params.early_stopping_epochs),
        SaveModelAndOptimizerCallback(fname=cfg.lognames.best_model_file),
        NaNDebugger(),
        # MixedPrecision(clip=0.1),
    ]
    # instantiate learner object
    learner = fastai.learner.Learner(
        dls,
        full_model,
        opt_func=opt_func,
        loss_func=combined_loss,
        metrics=metrics,
        cbs=callbacks,
        path=root_path,
        model_dir=cfg.paths.model_folder_save,
        splitter=get_param_groups,
        train_bn=True,
        lr=[
            cfg.optim_params.learning_rates.lr,
            cfg.optim_params.learning_rates.lr_backbone,
            cfg.optim_params.learning_rates.lr_linear_proj_mult,
        ],
    )

    return learner


class SaveModelAndOptimizerCallback(SaveModelCallback):
    "A `SaveModelCallback` that also saves the optimizer state"

    def after_epoch(self):
        "Compare the value monitored to its best score and maybe save the model and optimizer state"
        super().after_epoch()
        if self.new_best:
            self.learn.save(file=self.fname, with_opt=True)


class NaNDebugger(Callback):
    def after_batch(self):
        if torch.isnan(self.learn.loss) or torch.isinf(self.learn.loss):
            print(f"NaN/Inf detected in loss at batch {self.learn.iter}")
            for name, param in self.model.named_parameters():
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf detected in gradients of {name}")
