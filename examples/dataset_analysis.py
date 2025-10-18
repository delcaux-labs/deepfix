from deepfix.client import DeepFixClient

#env_file="D:/workspace/repos/deepfix/.env"
dataset_name="cafetaria-foodwaste-lstroetmann"

client = DeepFixClient(timeout=120)
result = client.diagnose_dataset(dataset_name=dataset_name)
print(result.to_text())