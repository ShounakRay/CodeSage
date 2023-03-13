from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
