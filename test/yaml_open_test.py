import yaml

with open("../lists/ucf-crime/anet1.3_labels.csv") as f:
    data = yaml.safe_load(f)

print(data)
