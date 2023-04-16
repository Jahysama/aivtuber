from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def build_model_and_tokenizer_for():
    base = "decapoda-research/llama-7b-hf",
    finetuned = "tloen/alpaca-lora-7b",
    tokenizer = LlamaTokenizer.from_pretrained(base)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    model = LlamaForCausalLM.from_pretrained(
        base,
        load_in_8bit=True,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, finetuned, device_map={'': 0})
    return model, tokenizer

def inference_fn(model,
                tokenizer,
                user_input,
                generation_settings,
                char_settings):

    if not generation_settings:
        generation_settings = GenerationConfig(
            temperature=0.7,
            top_p=0.75,
            num_beams=4,
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
    inputs = tokenizer(final_input, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_settings,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=256
    )
    output = ''
    for s in generation_output.sequences:
        output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()
