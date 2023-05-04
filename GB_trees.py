import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd

dataset = pd.read_csv("stroke-test.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="stroke")

tuner = tfdf.tuner.RandomSearch(num_trials=50)
tuner.choice("max_depth", [14,12,11,10,9,8,7,6,5,4,3])

model = tfdf.keras.GradientBoostedTreesModel(max_depth=14, selective_gradient_boosting_ratio=1.0, min_examples=4, shrinkage=0.11, growing_strategy="LOCAL",
                                             sampling_method="GOSS", goss_alpha=0.21, validation_ratio=0.08, missing_value_policy="LOCAL_IMPUTATION", categorical_algorithm="RANDOM")
model.fit(tf_dataset)

print(model.summary())


inputs = tf.keras.layers.Input(shape=(3,))
outputs = tf.keras.layers.Dense(2)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.metrics_names
print(model)
