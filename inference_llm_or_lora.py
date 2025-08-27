from unsloth import FastModel as FastLanguageModel
import argparse
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer, BatchEncoding
import torch

class Args:
	checkpoint: str

def SetupArguments() -> Args:
	args = argparse.ArgumentParser()
	args.add_argument("--checkpoint", type=str, required=True)
	return args.parse_args()

def ExtractPrompt(prompt: str):
	indexStart = prompt.find(' ')
	if indexStart == -1:
		if prompt.startswith('/'):
			return prompt.strip(), ""
		else:
			return "", prompt
	first = prompt[:indexStart].strip()
	second = prompt[indexStart+1:]
	if first.startswith('/'):
		return first.lower(), second
	return "", prompt

def Decode(model_inputs: BatchEncoding, generated_ids: torch.Tensor, tokenizer: Qwen2Tokenizer):
	output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

	# parsing thinking content
	try:
		# rindex finding 151668 (</think>)
		index = len(output_ids) - output_ids[::-1].index(151668)
	except ValueError:
		index = 0

	thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
	output_text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

	return thinking_content, output_text

def main(args: Args):
	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = args.checkpoint,
		max_seq_length = 2048,
		load_in_4bit = True,
		device_map="auto",
	)
	FastLanguageModel.for_inference(model)
	model: Qwen3ForCausalLM
	tokenizer: Qwen2Tokenizer

	print("- Type `/bye` to exit. Type `/system xxx` to set system prompt. Type `/clear` or `/clean` to clean history.")
	messages = []
	while True:
		input_str = input(">>> ")
		command, content = ExtractPrompt(input_str)
		if command == "/bye":
			break
		if command == "/system":
			if len(content.strip()) == 0:
				if len(messages) > 0 and messages[0]["role"] == "system":
					messages = messages[1:]
					print("- Clear system prompt.")
				else:
					print("- System prompt was not set.")
			else:
				if len(messages) == 0 or messages[0]["role"] != "system":
					messages = [{
						"role": "system",
						"content": "",
					}] + messages

				messages[0]["content"] = content
				print(f"- Set system prompt to `{content}`.")
			continue
		if command == "/clear" or command == "/clean":
			if len(messages) > 0 and messages[0]["role"] == "system":
				messages = [ messages[0] ]
			else:
				messages = []
				print("- Clean.")
			continue
		if len(command) != 0:
			print("- Unknown command: ", command)
			continue

		messages.append({
			"role": "user",
			"content": content,
		})

		text = tokenizer.apply_chat_template(
			messages,
			tokenize = False,
			add_generation_prompt = True, # Must add for generation
			enable_thinking = False, # Disable thinking
		)

		model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

		generated_ids = model.generate(
			**model_inputs,
			max_new_tokens = 256, # Increase for longer outputs!
			temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
		)

		thinking_content, output_text = Decode(model_inputs, generated_ids, tokenizer)

		print("thinking content:", thinking_content)
		print("content:", output_text)

		messages.append({
			"role": "assistant",
			"content": output_text,
		})

		print()

if __name__ == "__main__":
	args = SetupArguments()
	main(args)