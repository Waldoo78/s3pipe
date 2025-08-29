import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Optional, Tuple, Any, Dict, Iterable
from gluonts.core.component import validated
from gluonts.itertools import select
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.distributions import Output, StudentTOutput
from gluonts.torch.util import weighted_average
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import as_stacked_batches
from gluonts.itertools import Cyclic
from gluonts.torch.model.estimator import PyTorchLightningEstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
   AddObservedValuesIndicator,
   ExpectedNumInstanceSampler,
   InstanceSampler,
   InstanceSplitter,
   SelectFields,
   TestSplitSampler,
   Transformation,
   ValidationSplitSampler,
)


PREDICTION_INPUT_NAMES = [
   "past_target",
   "past_observed_values",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
   "future_target",
   "future_observed_values",
]

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) https://openreview.net/pdf?id=cGDAkQo1C0p
    https://github.com/ts-kim/RevIN
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class LinearRegressionModel(nn.Module):
   @validated()
   def __init__(
       self,
       prediction_length: int,
       context_length: int,
       scaling: str,
       hidden_dimensions: Optional[List[int]] = None,
       distr_output: Output = StudentTOutput(),
       batch_norm: bool = False,
       use_revin: str = "auto",
   ) -> None:
       super().__init__()

       self.prediction_length = prediction_length
       self.context_length = context_length
       self.distr_output = distr_output
       self.use_revin_mode = use_revin
       
       if scaling == "mean":
           self.scaler = MeanScaler(keepdim=True)
       elif scaling == "std":
           self.scaler = StdScaler(keepdim=True)
       else:
           self.scaler = NOPScaler(keepdim=True)

       self.hidden_dimensions = []
       self.nn = nn.Linear(context_length, prediction_length)
       self.args_proj = self.distr_output.get_args_proj(1)
       self.revin = None
       self.use_revin = False

   def _setup_revin(self, input_tensor):
       if input_tensor.dim() == 3:
           num_features = input_tensor.shape[-1]
       else:
           num_features = 1
           
       if self.use_revin_mode == "auto":
           self.use_revin = (num_features > 1)
       else:
           self.use_revin = bool(self.use_revin_mode)
           
       if self.use_revin and self.revin is None:
           self.revin = RevIN(num_features=num_features)

   def describe_inputs(self, batch_size=1) -> InputSpec:
       return InputSpec({
           "past_target": Input(
               shape=(batch_size, self.context_length), dtype=torch.float
           ),
           "past_observed_values": Input(
               shape=(batch_size, self.context_length), dtype=torch.float
           ),
       }, torch.zeros)

   def forward(
       self,
       past_target: torch.Tensor,
       past_observed_values: torch.Tensor,
   ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
       
       self._setup_revin(past_target)
       
       if past_target.dim() == 3:
           batch_size, seq_len, num_features = past_target.shape
           is_multivariate = True
       else:
           batch_size, seq_len = past_target.shape
           num_features = 1
           is_multivariate = False
           past_target = past_target.unsqueeze(-1)
           past_observed_values = past_observed_values.unsqueeze(-1)

       if is_multivariate:
           predictions = []
           locs = []
           scales = []
           
           for i in range(num_features):
               feature_target = past_target[:, :, i]
               feature_observed = past_observed_values[:, :, i]
               
               scaled_target, loc, scale = self.scaler(feature_target, feature_observed)
               
               if self.use_revin and self.revin is not None:
                   scaled_target_reshaped = scaled_target.unsqueeze(-1)
                   scaled_target_norm = self.revin(scaled_target_reshaped, mode='norm')
                   scaled_target_norm = scaled_target_norm.squeeze(-1)
               else:
                   scaled_target_norm = scaled_target
                   
               pred = self.nn(scaled_target_norm)
               
               if self.use_revin and self.revin is not None:
                   pred_reshaped = pred.unsqueeze(-1)
                   pred = self.revin(pred_reshaped, mode='denorm').squeeze(-1)
               
               predictions.append(pred.unsqueeze(-1))
               locs.append(loc)
               scales.append(scale)
           
           predictions = torch.cat(predictions, dim=-1)
           loc = torch.stack(locs, dim=-1)
           scale = torch.stack(scales, dim=-1)
           
       else:
           feature_target = past_target.squeeze(-1)
           feature_observed = past_observed_values.squeeze(-1)
           
           scaled_target, loc, scale = self.scaler(feature_target, feature_observed)
           
           if self.use_revin and self.revin is not None:
               scaled_target_reshaped = scaled_target.unsqueeze(-1)
               scaled_target_norm = self.revin(scaled_target_reshaped, mode='norm')
               scaled_target_norm = scaled_target_norm.squeeze(-1)
           else:
               scaled_target_norm = scaled_target
               
           predictions = self.nn(scaled_target_norm)
           
           if self.use_revin and self.revin is not None:
               predictions_reshaped = predictions.unsqueeze(-1)
               predictions = self.revin(predictions_reshaped, mode='denorm').squeeze(-1)
               
           predictions = predictions.unsqueeze(-1)

       distr_args = self.args_proj(predictions)
       
       return distr_args, loc, scale

   def loss(
       self,
       past_target: torch.Tensor,
       past_observed_values: torch.Tensor,
       future_target: torch.Tensor,
       future_observed_values: torch.Tensor,
   ) -> torch.Tensor:
       distr_args, loc, scale = self(
           past_target=past_target, 
           past_observed_values=past_observed_values
       )
       loss = self.distr_output.loss(
           target=future_target,
           distr_args=distr_args,
           loc=loc,
           scale=scale,
       )
       return weighted_average(loss, weights=future_observed_values, dim=-1)


class LinearRegressionLightningModule(pl.LightningModule):
   @validated()
   def __init__(
       self,
       model_kwargs: dict,
       lr: float = 1e-3,
       weight_decay: float = 1e-8,
   ):
       super().__init__()
       self.save_hyperparameters()
       self.model = LinearRegressionModel(**model_kwargs)
       self.lr = lr
       self.weight_decay = weight_decay
       self.inputs = self.model.describe_inputs()

   def forward(self, *args, **kwargs):
       return self.model.forward(*args, **kwargs)

   def training_step(self, batch, batch_idx: int):
       train_loss = self.model.loss(
           **select(self.inputs, batch),
           future_target=batch["future_target"],
           future_observed_values=batch["future_observed_values"],
       ).mean()

       self.log("train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True)
       return train_loss

   def validation_step(self, batch, batch_idx: int):
       val_loss = self.model.loss(
           **select(self.inputs, batch),
           future_target=batch["future_target"],
           future_observed_values=batch["future_observed_values"],
       ).mean()

       self.log("val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True)
       return val_loss

   def configure_optimizers(self):
       return torch.optim.Adam(
           self.model.parameters(),
           lr=self.lr,
           weight_decay=self.weight_decay,
       )


class LinearRegression(PyTorchLightningEstimator):
   @validated()
   def __init__(
       self,
       prediction_length: int,
       context_length: Optional[int] = None,
       hidden_dimensions: Optional[List[int]] = None,
       lr: float = 1e-3,
       weight_decay: float = 1e-5,
       distr_output: Output = StudentTOutput(),
       batch_norm: bool = False,
       batch_size: int = 32,
       num_batches_per_epoch: int = 50,
       scaling: str = "mean",
       use_revin: str = "auto",
       trainer_kwargs: Optional[Dict[str, Any]] = None,
       train_sampler: Optional[InstanceSampler] = None,
       validation_sampler: Optional[InstanceSampler] = None,
   ) -> None:
       default_trainer_kwargs = {
           "max_epochs": 100,
           "gradient_clip_val": 10.0,
       }
       if trainer_kwargs is not None:
           default_trainer_kwargs.update(trainer_kwargs)
       super().__init__(trainer_kwargs=default_trainer_kwargs)

       self.prediction_length = prediction_length
       self.context_length = context_length or 10 * prediction_length
       self.hidden_dimensions = []
       self.lr = lr
       self.weight_decay = weight_decay
       self.distr_output = distr_output
       self.batch_norm = batch_norm
       self.batch_size = batch_size
       self.num_batches_per_epoch = num_batches_per_epoch
       self.scaling = scaling
       self.use_revin = use_revin
       
       self.train_sampler = train_sampler or ExpectedNumInstanceSampler(
           num_instances=50.0, min_future=prediction_length
       )
       self.validation_sampler = validation_sampler or ValidationSplitSampler(
           min_future=prediction_length
       )

   def create_transformation(self) -> Transformation:
       return SelectFields([
           FieldName.ITEM_ID,
           FieldName.INFO,
           FieldName.START,
           FieldName.TARGET,
       ], allow_missing=True) + AddObservedValuesIndicator(
           target_field=FieldName.TARGET,
           output_field=FieldName.OBSERVED_VALUES,
       )

   def create_lightning_module(self) -> pl.LightningModule:
       return LinearRegressionLightningModule(
           lr=self.lr,
           weight_decay=self.weight_decay,
           model_kwargs={
               "prediction_length": self.prediction_length,
               "context_length": self.context_length,
               "hidden_dimensions": self.hidden_dimensions,
               "distr_output": self.distr_output,
               "batch_norm": self.batch_norm,
               "scaling": self.scaling,
               "use_revin": self.use_revin,
           },
       )

   def _create_instance_splitter(self, module, mode: str):
       assert mode in ["training", "validation", "test"]
       instance_sampler = {
           "training": self.train_sampler,
           "validation": self.validation_sampler,
           "test": TestSplitSampler(),
       }[mode]
       return InstanceSplitter(
           target_field=FieldName.TARGET,
           is_pad_field=FieldName.IS_PAD,
           start_field=FieldName.START,
           forecast_start_field=FieldName.FORECAST_START,
           instance_sampler=instance_sampler,
           past_length=self.context_length,
           future_length=self.prediction_length,
           time_series_fields=[FieldName.OBSERVED_VALUES],
           dummy_value=self.distr_output.value_in_support,
       )

   def create_training_data_loader(self, data: Dataset, module, **kwargs):
       data = Cyclic(data).stream()
       instances = self._create_instance_splitter(module, "training").apply(data, is_train=True)
       return as_stacked_batches(
           instances,
           batch_size=self.batch_size,
           field_names=TRAINING_INPUT_NAMES,
           output_type=torch.tensor,
           num_batches_per_epoch=self.num_batches_per_epoch,
       )

   def create_validation_data_loader(self, data: Dataset, module, **kwargs):
       instances = self._create_instance_splitter(module, "validation").apply(data, is_train=True)
       return as_stacked_batches(
           instances,
           batch_size=self.batch_size,
           field_names=TRAINING_INPUT_NAMES,
           output_type=torch.tensor,
       )

   def create_predictor(self, transformation: Transformation, module):
       prediction_splitter = self._create_instance_splitter(module, "test")
       return PyTorchPredictor(
           input_transform=transformation + prediction_splitter,
           input_names=PREDICTION_INPUT_NAMES,
           prediction_net=module,
           forecast_generator=self.distr_output.forecast_generator,
           batch_size=self.batch_size,
           prediction_length=self.prediction_length,
           device="auto",
       )