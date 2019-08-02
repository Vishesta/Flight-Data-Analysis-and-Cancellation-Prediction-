DATA_DIR = "data"
PROCESSED_DIR = "processed"
FEATURES = ["MONTH","DAY_OF_WEEK","AIRLINE","SCHEDULED_DEPARTURE",
            "SCHEDULED_ARRIVAL","DISTANCE", "SCHEDULED_TIME",
            "ORIGIN_LATITUDE","ORIGIN_LONGITUDE",
            "DESTINATION_LATITUDE","DESTINATION_LONGITUDE"]
PREDICTOR = ["CANCELLATION_REASON"]
reason = "B"
test_size = 0.3
# categorical features 
dummy_features = ["AIRLINE","MONTH","DAY_OF_WEEK"]
# non-categorical features
scale_features = [item for item in FEATURES if item not in set(dummy_features)]
