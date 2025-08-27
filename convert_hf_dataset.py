from datasets import load_dataset
import json
import argparse
import typing
import os
import re

class Args:
	hf_repo: str
	output: str
	dataset_format: str

	instruction_text_field: str
	input_text_field: str
	output_text_field: str

	FORMAT_ALPACA = "alpaca"
	FORMAT_SHAREGPT = "sharegpt"
	FORMAT_DPO = "dpo"

def SetupArgs() -> Args:
	parser = argparse.ArgumentParser()
	parser.add_argument("--hf_repo", type=str, required=True)
	parser.add_argument("--output", type=str, required=True, help="Output filename like `data/dataset/llm/custom_dataset.json`")
	parser.add_argument("--dataset_format", type=str, choices=[Args.FORMAT_ALPACA, Args.FORMAT_SHAREGPT, Args.FORMAT_DPO], required=True)
	parser.add_argument("--instruction_text_field", type=str, default="instruction", help="leave it empty to ignore this field")
	parser.add_argument("--input_text_field", type=str, default="input")
	parser.add_argument("--output_text_field", type=str, default="output")
	return parser.parse_args()

def postprocess(session_history: typing.List[typing.Dict[str, typing.Union[str, typing.List[typing.Dict[str,typing.Any]]]]]) -> typing.Optional[typing.List[typing.Dict[str, typing.Union[str, typing.List[typing.Dict[str,typing.Any]]]]]]:
	"""
	Write your own postprocessing function here. Return None to skip this item.
	"""
	pattern = re.compile(r'[?？]\s*[\U0001F300-\U0001FAFF]') # fileter: xxx?😄xxx
	new_session: typing.List[typing.Dict[str, typing.Union[str, typing.List[typing.Dict[str,typing.Any]]]]] = []
	for item in session_history:
		if item["role"] == "user" and item["content"][0]["text"].strip() == "":
			return None
		if item["role"] == "system":
			continue
		if item["role"] == "assistant" and len(item["content"][0]["text"]) < 50:
			return None
		if item["role"] == "assistant" and bool(pattern.search(item["content"][0]["text"])):
			return None
		# if item["role"] == "assistant" and len(item["content"][0]["text"].split(' ')) < 50: # english dataset only
		# 	return None
		new_session.append(item)
	return new_session

def main(args: Args):
	# 1. 加载 Hugging Face 上的数据集
	dataset = load_dataset(args.hf_repo, split="train")
	print("Loading dataset:", dataset)

	# 2. 格式化数据集
	flatten: typing.List[typing.Dict[str, typing.List[typing.Dict[str, typing.List[typing.Dict[str, typing.Any]]]]]] = []
	if args.dataset_format == Args.FORMAT_ALPACA or args.dataset_format == Args.FORMAT_DPO:
		for item in dataset.to_list():
			target: typing.List[typing.Dict[str, typing.Union[str, typing.List[typing.Dict[str,typing.Any]]]]] = []

			if args.instruction_text_field != "" and args.instruction_text_field in item:
				if (item[args.instruction_text_field] is None or len(item[args.instruction_text_field].strip()) == 0) and len(item[args.input_text_field].strip())== 0:
					continue
			else:
				if len(item[args.input_text_field].strip()) == 0:
					continue
			
			if args.instruction_text_field != "" and args.instruction_text_field in item:
				target.append({
					"role": "system",
					"content": [
						{ "type": "text", "text": item[args.instruction_text_field] },
					]
				})
			target.append({
				"role": "user",
				"content": [
					{ "type": "text", "text": item[args.input_text_field] },
				]
			})
			target.append({
				"role": "assistant",
				"content": [
					{ "type": "text", "text": item[args.output_text_field] },
				]
			})

			new_item = postprocess(target)
			if new_item is not None:
				flatten.append({
					"messages": new_item
				})
	elif args.dataset_format == Args.FORMAT_SHAREGPT:
		for item in dataset.to_list():
			new_item = postprocess(dataset["messages"])
			if new_item is None:
				flatten.append({
					"messages": new_item
				})
	else:
		raise ValueError(f"`{args.dataset_format}` is not supported.")
	
	# 3. 保存数据集
	outputDir = os.path.dirname(args.output)
	os.makedirs(outputDir, exist_ok=True)
	if args.output.endswith(".jsonl"):
		with open(args.output, "w", encoding="utf-8") as f:
			for item in flatten:
				f.write(json.dumps(item, ensure_ascii=False) + "\n")
	elif args.output.endswith(".json"):
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(json.dumps(flatten, indent=4, ensure_ascii=False))
	
	print(f"- Dataset items: {len(flatten)}")
	print(f"- Save to {args.output}")

if __name__ == "__main__":
	args = SetupArgs()
	main(args)