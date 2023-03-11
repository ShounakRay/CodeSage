from transformers import T5ForConditionalGeneration, RobertaTokenizer, pipeline

model = T5ForConditionalGeneration.from_pretrained("joetey/glued_code_to_code_model")
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

translator = pipeline("translation", model=model, tokenizer=tokenizer)
text = "def hello_world():\nprint(str)"
output = translator(text)

print(output[0]["translation_text"])