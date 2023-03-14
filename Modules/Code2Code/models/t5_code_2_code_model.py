from Modules.Code2Code.models.base_model import BaseCode2CodeModel
from datasets import Dataset
from transformers import RobertaTokenizer, pipeline, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np

class T5Code2CodeModel(BaseCode2CodeModel):
    pretrained_model_name = "Salesforce/codet5-"
    def __init__(
        self, 
        model_size: str, 
        max_length=512,     
        truncation=True,
        metric="chrf" 
            ):
        """Init salesforce codet5 model of a given size

        Args:
            model_size (str): one of "large", "small", "base"
        """
        super().__init__()
        self.pretrained_model_name += model_size
        self.metric = evaluate.load(metric)
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        self.max_length = max_length
        self.truncation = truncation
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_name)
        self.finetuned_model_name = self.pretrained_model_name
    
    def preprocess_function(self, examples: Dataset):
        
        inputs = [self.prefix + example for example in examples["input"]]
        outputs = [example for example in examples["target"]]
        # max_cluster_amount = 50
        # tokenized_input_references_pairs = {"input_ids": [], "labels": []}
        # for i in range(len(inputs)):
        #     references = outputs[i] + max(0, max_cluster_amount - len(outputs[i])) * ["BLEH"]
        #     encodings_i = self.tokenizer(inputs[i], text_target=references, max_length=512, truncation=True)
        #     tokenized_input_references_pairs["input_ids"].append(encodings_i["input_ids"])
        #     tokenized_input_references_pairs["labels"].append(encodings_i["labels"])
        # print(tokenized_input_references_pairs)
        tokenized_input_references_pairs = self.tokenizer(inputs, text_target=outputs, max_length=self.max_length, truncation=True)
        return tokenized_input_references_pairs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    
    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"chrf": result["score"]}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    
    def train(
        self,
        dataset: Dataset,
        output_model_dir: str,
        test_size=0.2,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        num_train_epochs=2,
    ):
        dataset = dataset.map(self.preprocess_function, batched=True, num_proc=4).train_test_split(test_size=test_size)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.pretrained_model)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_model_dir,
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            save_total_limit=num_train_epochs + 1,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
        )
        
        trainer = Seq2SeqTrainer(
            model=self.pretrained_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        self.finetuned_model_name = output_model_dir
        try:
            self.pretrained_model.push_to_hub(self.finetuned_model_name)
        except:
            print("Error pushing automatically to hub. Push manually via Python REPL.")
    
    def __call__(self, input_bad_code: str):
        text = self.prefix + input_bad_code
        model = T5ForConditionalGeneration.from_pretrained(self.finetuned_model_name)
        tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        return translator(text)