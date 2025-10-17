from deepfix.core.agents import DatasetAnalyzer

env_file="D:/workspace/repos/deepfix/.env"
dataset_name="cafetaria-foodwaste-lstroetmann"

analyzer = DatasetAnalyzer(env_file=env_file,)
result = analyzer.run(dataset_name=dataset_name)
