
from voyage_analytics.pipline.training_pipeline import TrainPipeline
model_trained_artifact:object=None
local_estimator:object=None 




obj = TrainPipeline()
model_trained_artifact = obj.run_pipeline()  # Only once at startup

print(model_trained_artifact)

ModelTrainerArtifact(model_trainer_artifact_regression=ModelTrainerArtifactRegression(trained_model_file_path='artifact\\01_21_2026_02_32_38\\model_trainer\\trained_model\\regression\\model.pkl', metric_artifact=RegressionMetricArtifact(r2_score=0.9999918658795588, mse=0.9838980104384437, root_mean_squared_error=np.float64(0.9919163323781113), mae=0.7181338613081975)), model_trainer_artifact_classification=ModelTrainerArtifactClassification(trained_model_file_path='artifact\\01_21_2026_02_32_38\\model_trainer\\trained_model\\classification\\model.pkl', metric_artifact=ClassificationMetricArtifact(accuracy_score=0.8582089552238806, precision_score=0.8609217426381606, recall_score=0.8582089552238806, f1_score=0.8578285634203229)), model_trainer_artifact_recumendation=ModelTrainerArtifactRecomendation(trained_model_file_path='artifact\\01_21_2026_02_32_38\\model_trainer\\trained_model\\recumendation\\model.pkl', metric_artifact=None))