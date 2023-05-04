import tensorflow_decision_forests as tfdf
import pandas as pd

dataset = pd.read_csv("stroke-train.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="stroke")
sampler = tfdf.tuner.dict_sampler.DiscreteDictSampler({ "max_depth": [4, 5, 6, 7]})
#tuner = tfdf.tuner.RandomSearch(sampler, num_trials=20)



# Hyper-parameters to optimize.
tuner = tfdf.keras.tuner.RandomSearch(
    tfdf.keras.GradientBoostedTreesModel(),
    sampler=sampler,
    objective="binary_crossentropy",
    max_trials=20
)

model = tfdf.keras.GradientBoostedTreesModel(tuner=tuner)
model.fit(tf_dataset)

print(model.summary())



dataset = pd.read_csv("stroke-train.csv")
tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="stroke")

sampler = tfdf.tuner.dict_sampler.DiscreteDictSampler({
    "max_depth": [4, 5, 6, 7]
})

tuner = tfdf.keras.tuner.RandomSearch(
    tfdf.keras.GradientBoostedTreesModel(),
    sampler=sampler,
    objective="binary_crossentropy",
    max_trials=20
)

tuner.search(tf_dataset)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

print(model.summary())