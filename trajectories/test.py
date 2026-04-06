import gzip, pickle

with gzip.open("PST_V2G_ProfixMax_25_real_optimal_25_500.pkl.gz", "rb") as f:
    trajs = pickle.load(f)

with gzip.open("PST_V2G_ProfixMax_25_real_optimal_25_450.pkl.gz", "wb") as f:
    pickle.dump(trajs[:450], f)