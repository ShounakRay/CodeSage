from datasets import load_dataset

dataset = load_dataset("michaelnath/annotated_github_dataset_2", split='train')

filtered_description = list(filter(None, dataset['detailed_description']))
filtered_purpose = list(filter(None, dataset['purpose']))
filtered_codetrans = list(filter(None, dataset['code_trans']))

print(len(filtered_codetrans))