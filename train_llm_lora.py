#Ref: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(14B)-Reasoning-Conversational.ipynb#scrollTo=TCv4vXHd61i7

from unsloth import FastLanguageModel, FastVisionModel
import json, os
from datasets import Dataset
import pandas as pd
from trl import SFTTrainer, SFTConfig  # 或者使用 Hugging Face Trainer
from transformers import Qwen2Tokenizer, Qwen2ForCausalLM
from unsloth.trainer import UnslothVisionDataCollator
import argparse
import utils
import typing
from PIL import Image

class Args:
	train_dir: str
	dataset: str
	checkpoint: str
	batch_size: int
	gradient_accumulation_steps: int
	max_steps: int
	learning_rate: float
	logging_steps: int
	seed: int
	lora_rank: int
	lora_alpha: int
	lora_dropout: float
	max_seq_length: int
	inference_after_train: bool

	vision_model: bool
	vision_train_only_language_model: bool

def SetupArgs() -> Args:
	args = argparse.ArgumentParser()
	args.add_argument("--train_dir", type=str, default="./train")
	args.add_argument("--dataset", type=str, default="./data/dataset/llm/dataset_info.json")
	args.add_argument("--checkpoint", type=str, required=True)
	args.add_argument("--batch_size", type=int, default=16)
	args.add_argument("--gradient_accumulation_steps", type=int, default=1)
	args.add_argument("--max_steps", type=int, default=100)
	args.add_argument("--learning_rate", type=float, default=2e-5)
	args.add_argument("--logging_steps", type=int, default=1)
	args.add_argument("--seed", type=int, default=3407)
	args.add_argument("--lora_rank", type=int, default=16, help="Choose any number > 0! Suggested 8, 16, 32, 64, 128")
	args.add_argument("--lora_alpha", type=int, default=16, help="Best to choose alpha = rank or rank*2")
	args.add_argument("--lora_dropout", type=float, default=0, help="Supports any, but = 0 is optimized")
	args.add_argument("--max_seq_length", type=int, default=2048, help="Context length - can be longer, but uses more memory")
	args.add_argument("--inference_after_train", action="store_true")

	args.add_argument("--vision_model", action="store_true", help="Turn it on if the checkpoint is a vision model")
	args.add_argument("--vision_train_only_text", action="store_true", help="Train only text model in vision model")
	return args.parse_args() # type: ignore

def loadDataset(datasetInfoFilename: str):
	basedir = os.path.dirname(datasetInfoFilename)
	with open(datasetInfoFilename, "r", encoding="utf-8") as f:
		datasetInfos: typing.Dict[str, typing.Dict[str, typing.Any]] = json.loads(f.read())

	conversations: typing.List[typing.List[typing.Dict[str, str]]] = []
	for datasetName, datasetInfo in datasetInfos.items():
		if not datasetInfo.get("enable", True):
			continue
		print(f"- Reading dataset {datasetName}")
		datasetFilename = os.path.join(basedir, datasetInfo["filename"])
		repeat_count = int(datasetInfo.get("repeat_count", 1))
		if not os.path.exists(datasetFilename):
			raise RuntimeError(f"- Dataset {datasetName} not found at {datasetFilename}")
		with open(datasetFilename, "r", encoding="utf-8") as f:
			dataset = json.loads(f.read())
		for item in dataset:
			if "messages" not in item:
				raise RuntimeError(f"- Dataset {datasetName} in sharegpt format should has `messages` key.")
			session: typing.List[typing.Dict[str, typing.Union[str, typing.List[typing.Dict[str, typing.Any]]]]] = item["messages"]
			new_session: typing.List[typing.Dict[str, typing.Any]] = []
			for conv in session:
				for subconv in conv["content"]:
					if subconv["type"] == "image":
						raise RuntimeError("Unsupported image content currently.")
						url = subconv["image"]
						img = Image.open(url)
					if subconv["type"] == "text":
						new_session.append({
							"role": conv["role"],
							"content": subconv["text"]
						})

			for i in range(repeat_count):
				conversations.append(new_session)
	return conversations

def main(args: Args):
	# 1. 设置工程文件夹
	expDir = utils.SetupExpDir(args.train_dir)
	print("Exp dir: ", expDir)

	# 2. 加载数据
	conversations = loadDataset(args.dataset)
	print(f"Dataset items: {len(conversations)}")

	# 3. 加载模型 via Unsloth
	cls = FastVisionModel if args.vision_model else FastLanguageModel
	model, tokenizer = cls.from_pretrained(
		model_name=args.checkpoint,
		load_in_4bit=True,
	)
	model: Qwen2ForCausalLM
	tokenizer: Qwen2Tokenizer

	# 4. 添加 LoRA/QLoRA adapter
	peft_config = {
		"r": args.lora_rank,
		"lora_alpha": args.lora_alpha,
		"lora_dropout": args.lora_dropout,
		"bias": "none",
		"use_gradient_checkpointing": "unsloth",
		"max_seq_length": args.max_seq_length
	}
	if args.vision_model:
		peft_config["finetune_vision_layers"] = not args.vision_train_only_language_model
		peft_config["finetune_language_layers"] = True
		peft_config["finetune_attention_modules"] = True
		peft_config["finetune_mlp_modules"] = True
	else:
		peft_config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	model = cls.get_peft_model(model, **peft_config)

	# 5. Tokenize 数据
	dataset_column: typing.List[str] = tokenizer.apply_chat_template(conversations, tokenize = False)
	print("- Preview first data:", dataset_column[0])
	trained_dataset = Dataset.from_dict({
		"text": dataset_column
	})
	trained_dataset = trained_dataset.shuffle(seed=42).flatten_indices()

	# 6. 设置训练参数
	training_args = SFTConfig(
		output_dir=expDir,
		dataset_text_field = "text",
		# dataset_kwargs={
		# 	"skip_prepare_dataset": True
		# },
		per_device_train_batch_size = args.batch_size,
		gradient_accumulation_steps = args.gradient_accumulation_steps, # Use GA to mimic batch size!
		warmup_steps = 5,
		# num_train_epochs = 1, # Set this for 1 full training run.
		max_steps = args.max_steps,
		learning_rate = args.learning_rate, # Reduce to 2e-5 for long training runs
		logging_steps = args.logging_steps,
		optim = "adamw_8bit",
		weight_decay = 0.01,
		lr_scheduler_type = "linear",
		seed = args.seed,
		report_to = "tensorboard", # Use this for WandB etc
		logging_dir=os.path.join(expDir, "logs"),
		max_seq_length=args.max_seq_length,
		save_steps=100,
	)

	# 7. 使用 SFTTrainer（TRL）进行训练
	trainer = SFTTrainer(
		model=model,
		train_dataset=trained_dataset,
		# data_collator=UnslothVisionDataCollator(model, tokenizer),
		eval_dataset=None,
		tokenizer=tokenizer,
		args=training_args,
	)

	# 8. 开始训练
	trainer_stats = trainer.train()

	lora_dir = os.path.join(expDir, "lora_model")
	model.save_pretrained(lora_dir)  # Local saving
	tokenizer.save_pretrained(lora_dir)

	if args.inference_after_train:
		from inference_llm_or_lora import Decode
		print("- Infer.")
		
		conversation = conversations[0]
		text = tokenizer.apply_chat_template(
			conversation,
			tokenize = False,
			add_generation_prompt = True, # Must add for generation
			enable_thinking = False, # Disable thinking
		)

		model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

		generated_ids = model.generate(
			**model_inputs,
			max_new_tokens = 256, # Increase for longer outputs!
			temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
		)

		thinking_content, output_text = Decode(model_inputs, generated_ids, tokenizer)

		for item in conversation:
			if item["role"] == "system":
				print(f"🚩 {item['content']['text']}")
			if item["role"] == "user":
				print(f"👨 {item['content']['text']}")
			if item["role"] == "assistant":
				print(f"💻 {item['content']['text']}")
		print(f"💭 {thinking_content}")
		print(f"🤖 {output_text}")

	print("- Done.")

if __name__ == "__main__":
	args = SetupArgs()
	main(args)