
from voyage_analytics.pipline.training_pipeline import TrainPipeline
model_trained_artifact:object=None
local_estimator:object=None 




obj = TrainPipeline()
model_trained_artifact = obj.run_pipeline()  # Only once at startup

print(model_trained_artifact)
