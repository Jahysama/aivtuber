from peft import PeftModel
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)


def build_model_and_tokenizer_for():
    base = "decapoda-research/llama-7b-hf"
    finetuned = "tloen/alpaca-lora-7b"
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model_base = LlamaForCausalLM.from_pretrained(
        base,
        quantization_config=quantization_config,
        device_map="auto",
        load_in_8bit=False,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(model_base, finetuned, torch_dtype=torch.float16, force_download=True)
    return model, tokenizer

def inference_fn(model,
                tokenizer,
                user_input,
                generation_settings,
                char_settings):

    if not generation_settings:
        generation_settings = GenerationConfig(
            temperature=0.3,
            top_p=0.75,
            num_beams=1,
            max_new_tokens=64,
        )

    if user_input:
        final_input = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {char_settings}

    ### Input:
    {user_input}

    ### Response:"""
    else:
        final_input = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {char_settings}

    ### Response:"""
    model.eval()
    inputs = tokenizer(final_input, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_settings,
            return_dict_in_generate=True,
            output_scores=True
        )
    output = ''
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()
