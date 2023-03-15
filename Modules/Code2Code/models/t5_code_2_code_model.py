from Modules.Code2Code.models.base_model import BaseCode2CodeModel
import multiprocessing
import json
from datasets import Dataset
from transformers import RobertaTokenizer, pipeline, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import shutil
import os


class T5Code2CodeModel(BaseCode2CodeModel):
    pretrained_model_name = "Salesforce/codet5-"
    def __init__(
        self, 
        model_size: str, 
        max_length=512,     
        truncation=True):
        """Init salesforce codet5 model of a given size

        Args:
            model_size (str): one of "large", "small", "base"
        """
        super().__init__()
        self.pretrained_model_name += model_size
        self.chrf_metric = evaluate.load('chrf')
        self.codebleu_metric = evaluate.load('dvitel/codebleu')
        self.bleu_metric = evaluate.load('sacrebleu')
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        self.max_length = max_length
        self.truncation = truncation
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_name)
        self.finetuned_model_name = self.pretrained_model_name
    
    def preprocess_function(self, examples: Dataset):
        
        inputs = [self.prefix + example for example in examples["input"]]
        outputs = [example for example in examples["target"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, padding="max_length", truncation=True)
        labels = self.tokenizer(outputs, max_length=self.max_length, padding="max_length", truncation=True).input_ids
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        model_inputs["labels"] = labels_with_ignore_index
        return model_inputs

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
        chrf_result = self.chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
        codebleu_result = self.codebleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"chrf": chrf_result["score"], "codebleu": codebleu_result, "bleu": bleu_result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    def train(
        self,
        dataset: Dataset,
        output_model_dir: str,
        C2C_TEST_SIZE=0.2,
        C2C_LR=2e-5,
        C2C_BATCH_SIZE=4,
        C2C_WEIGHT_DECAY=0.01,
        C2C_EPOCH_N=2,
    ):
        print("C2C (Preprocessing): Creating Bad Code -> Good Code Dataset")
        dataset = dataset.map(self.preprocess_function, batched=True).train_test_split(test_size=C2C_TEST_SIZE)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.pretrained_model)
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_model_dir,
            evaluation_strategy="epoch",
            learning_rate=C2C_LR,
            per_device_train_batch_size=C2C_BATCH_SIZE,
            per_device_eval_batch_size=C2C_BATCH_SIZE,
            weight_decay=C2C_WEIGHT_DECAY,
            save_total_limit=C2C_EPOCH_N + 1,
            num_train_epochs=C2C_EPOCH_N,
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
        print("C2C (Finetuning): Finetuning Model") 
        trainer.train()
        print(trainer.evaluate())
        with open(f"results/{output_model_dir}.json", "w+") as f:
            f.write(json.dumps(trainer.evaluate(), indent=2))
        self.finetuned_model_name = output_model_dir
        print("C2C (Finetuning): Finetuning Finished") 
        # try:
        #     self.pretrained_model.push_to_hub(self.finetuned_model_name)
        # except:
        #     print("Error pushing automatically to hub. Push manually via Python REPL.")

        if os.path.exists(output_model_dir):
            try:
                shutil.rmtree(output_model_dir)
                print(f"Directory '{output_model_dir}' has been deleted.")
            except OSError as e:
                print(f"Error deleting directory '{output_model_dir}': {e}")
        else:
            print(f"Directory '{output_model_dir}' does not exist.")
        
    
    def __call__(self, input_bad_code: str):
        text = self.prefix + input_bad_code
        model = T5ForConditionalGeneration.from_pretrained(self.finetuned_model_name)
        tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        return translator(text)