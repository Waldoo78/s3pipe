import json
import os 
import torch
import gc
import csv
import optuna

from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gift_eval.data import Dataset
from gluonts.ev.metrics import (
 MSE, MAE, MASE, MAPE, SMAPE, MSIS,
 RMSE, NRMSE, ND, MeanWeightedSumQuantileLoss
)
from gluonts.transform import ExpectedNumInstanceSampler

from lightning.pytorch.callbacks import EarlyStopping, Callback
from linear_regressor.linear_regression import LinearRegression
from dotenv import load_dotenv

gift_path = "/home/wbouainouche/Samformer_distillation/gift_eval/GiftEval"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['DATA_PATH'] = gift_path
os.environ['GIFT_EVAL_DATA_PATH'] = gift_path

short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

All_datasets = list(set(short_datasets.split() + med_long_datasets.split()))

dataset_properties_map = json.load(open(os.path.join("/home/wbouainouche/Samformer_distillation/gift_eval/dataset_properties.json")))

metrics = [
   MSE(forecast_type="mean"), MSE(forecast_type=0.5),
   MAE(), MASE(), MAPE(), SMAPE(), MSIS(),
   RMSE(), NRMSE(), ND(),
   MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
]

pretty_names = {
   "saugeenday": "saugeen",
   "temperature_rain_with_missing": "temperature_rain",
   "kdd_cup_2018_with_missing": "kdd_cup_2018",
   "car_parts_with_missing": "car_parts",
}

output_dir = "/home/wbouainouche/Samformer_distillation/GIFT-Eval/results/LinearRegressionRevIN"
os.makedirs(output_dir, exist_ok=True)

models_dir = os.path.join(output_dir, "models")
configs_dir = os.path.join(output_dir, "configs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(configs_dir, exist_ok=True)

csv_file = os.path.join(output_dir, "all_results.csv")

if not os.path.exists(csv_file):
   with open(csv_file, "w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow([
           "dataset", "model", "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
           "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]",
           "eval_metrics/sMAPE[0.5]", "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
           "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
           "eval_metrics/mean_weighted_sum_quantile_loss", "domain", "num_variates",
       ])

#Fixed parameters 
PARAMS = {
   'num_instances': 100.0,  
   'max_epochs': 150,
   'patience': 5
}


n_trials = 20

class ValidationLossTracker(Callback):
   def __init__(self):
       self.val_losses = []
       self.best_val_loss = float('inf')
   
   def on_validation_epoch_end(self, trainer, pl_module):
       val_loss = trainer.callback_metrics.get('val_loss')
       if val_loss is not None:
           current_loss = val_loss.item()
           self.val_losses.append(current_loss)
           if current_loss < self.best_val_loss:
               self.best_val_loss = current_loss

class LinearRegressionObjective:
   def __init__(self, dataset, ds_config, season_length):
       self.dataset = dataset
       self.ds_config = ds_config
       self.season_length = season_length

   def get_params(self, trial):
       return {
           "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
           "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
           "context_length_multiplier": trial.suggest_categorical("context_length_multiplier", [2,5,10,15,20]),
       }
   
   def __call__(self, trial):
       try:
           params = self.get_params(trial)
           val_tracker = ValidationLossTracker()
           context_length = params["context_length_multiplier"] * self.dataset.prediction_length
           
           estimator = LinearRegression(
               prediction_length=self.dataset.prediction_length,
               context_length=context_length,
               use_revin="auto",
               train_sampler=ExpectedNumInstanceSampler(
                   num_instances=PARAMS['num_instances'], 
                   min_future=self.dataset.prediction_length
               ),
               scaling='std',
               lr=params['lr'],
               weight_decay=params['weight_decay'],
               batch_size=1024,
               num_batches_per_epoch=100,
               trainer_kwargs=dict(
                   max_epochs=15,
                   gradient_clip_val=None,
                   callbacks=[
                       val_tracker
                   ],
                   enable_progress_bar=False,
                   enable_model_summary=False,
                   logger=False,
               ),
           )
           
           predictor = estimator.train(self.dataset.training_dataset, self.dataset.validation_dataset)
           val_loss = val_tracker.best_val_loss
           
           del predictor, estimator
           gc.collect()
           torch.cuda.empty_cache()
           
           return val_loss
           
       except Exception as e:
           print(f"Trial failed: {e}")
           return float('inf')

for x in All_datasets:
   for term in ["short", "medium", "long"]:
       if (term in ["medium", "long"]) and x not in med_long_datasets.split():
           continue
           
       all_datasets = [x]
       
       for ds_name in all_datasets:
           print(f"Processing {ds_name} - {term}")

           if "/" in ds_name:
               ds_key = ds_name.split("/")[0].lower()
               ds_freq = ds_name.split("/")[1]
           else:
               ds_key = ds_name.lower()
               ds_key = pretty_names.get(ds_key, ds_key)
               ds_freq = dataset_properties_map[ds_key]["frequency"]
           
           ds_key = pretty_names.get(ds_key, ds_key)
           ds_config = f"{ds_key}/{ds_freq}/{term}"
           
           safe_name = ds_config.replace("/", "_")
           model_path = os.path.join(models_dir, f"{safe_name}.pkl")
           config_path = os.path.join(configs_dir, f"{safe_name}_config.json")

           try:
               univariate = (Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1)
               to_univariate = not univariate
               dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
               season_length = get_seasonality(dataset.freq)
               
               study = optuna.create_study(direction="minimize")
               objective = LinearRegressionObjective(dataset, ds_config, season_length)
               study.optimize(objective, n_trials=n_trials)
               
               best_params = study.best_params

               best_context_length = best_params['context_length_multiplier'] * dataset.prediction_length
               
               estimator = LinearRegression(
                   prediction_length=dataset.prediction_length,
                   context_length=best_context_length,
                   use_revin="auto",
                   train_sampler=ExpectedNumInstanceSampler(
                       num_instances=PARAMS['num_instances'], 
                       min_future=dataset.prediction_length
                   ),
                   scaling='std',
                   lr=best_params['lr'],
                   weight_decay=best_params['weight_decay'],
                   batch_size=1024,
                   num_batches_per_epoch=100,
                   trainer_kwargs=dict(
                       max_epochs=PARAMS['max_epochs'],
                       gradient_clip_val=None,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=PARAMS['patience'])],
                       logger=False,
                   ),
               )
               
               predictor = estimator.train(dataset.training_dataset, dataset.validation_dataset)
               res = evaluate_model(predictor, test_data=dataset.test_data, metrics=metrics, batch_size=2048, seasonality=season_length)
               
               torch.save(predictor, model_path)
               
               config_data = {
                   "dataset": ds_config,
                   "dataset_info": {
                       "prediction_length": dataset.prediction_length,
                       "context_length": best_context_length,
                       "frequency": dataset.freq,
                       "target_dim": dataset.target_dim,
                       "to_univariate": to_univariate
                   },
                   "optuna_optimization": {
                       "n_trials": n_trials,
                       "best_params": best_params,
                       "best_value": study.best_value
                   },
                   "fixed_hyperparameters": PARAMS,
                   "architecture": {
                       "model_type": "LinearRegressionRevIN",
                       "use_revin": "auto",
                       "hidden_dimensions": []
                   },
                   "final_metrics": {
                       "MASE": res["MASE[0.5]"][0],
                       "MSE": res["MSE[0.5]"][0],
                       "MAE": res["MAE[0.5]"][0],
                       "RMSE": res["RMSE[mean]"][0]
                   },
                   "model_path": model_path
               }
               
               with open(config_path, "w") as f:
                   json.dump(config_data, f, indent=2)
               
               with open(csv_file, "a", newline="") as f:
                   writer = csv.writer(f)
                   writer.writerow([
                       ds_config, "LinearRegressionRevIN",
                       res["MSE[mean]"][0], res["MSE[0.5]"][0], res["MAE[0.5]"][0],
                       res["MASE[0.5]"][0], res["MAPE[0.5]"][0], res["sMAPE[0.5]"][0],
                       res["MSIS"][0], res["RMSE[mean]"][0], res["NRMSE[mean]"][0],
                       res["ND[0.5]"][0], res["mean_weighted_sum_quantile_loss"][0],
                       dataset_properties_map[ds_key]["domain"],
                       dataset_properties_map[ds_key]["num_variates"],
                   ])
               
               print(f"{ds_config}: MASE={res['MASE[0.5]'][0]:.3f}")
               
               del predictor, estimator, dataset, res, study, objective
               gc.collect()
               torch.cuda.empty_cache()
               
           except Exception as e:
               print(f"Failed for {ds_config}: {e}")