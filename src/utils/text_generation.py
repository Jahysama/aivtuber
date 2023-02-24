import logging
import typing as t

import torch
import transformers
import re

logger = logging.getLogger(__name__)
BAD_CHARS_FOR_REGEX_REGEX = re.compile(r"[-\/\\^$*+?.()|[\]{}]")


def build_model_and_tokenizer_for(
    model_name: str
) -> t.Tuple[transformers.AutoModelForCausalLM, transformers.AutoTokenizer]:
    '''Sets up the model and accompanying objects.'''
    logger.info(f"Loading tokenizer for {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # NOTE(11b): non-OPT models support passing this in at inference time, might
    # be worth refactoring for a debug version so we're able to experiment on
    # the fly
    bad_words_ids = [
        tokenizer(bad_word, add_special_tokens=False).input_ids
        for bad_word in _build_bad_words_list_for(model_name)
    ]

    logger.info(f"Loading the {model_name} model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, bad_words_ids=bad_words_ids)
    model.eval().half().to("cuda")

    logger.info("Model and tokenizer are ready")
    return model, tokenizer


def run_raw_inference(model: transformers.AutoModelForCausalLM,
                      tokenizer: transformers.AutoTokenizer, prompt: str,
                      user_message: str, **kwargs: t.Any) -> str:
    '''
    Runs inference on the model, and attempts to returns only the newly
    generated text.

    :param model: Model to perform inference with.
    :param tokenizer: Tokenizer to tokenize input with.
    :param prompt: Input to feed to the model.
    :param user_message: The user's raw message, exactly as appended to the end
        of `prompt`. Used for trimming the original input from the model output.
    :return: Decoded model generation.
    '''
    tokenized_items = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Atrocious code to stop generation when the model outputs "\nYou: " in
    # freshly generated text. Feel free to send in a PR if you know of a
    # cleaner way to do this.
    stopping_criteria_list = transformers.StoppingCriteriaList([
        _SentinelTokenStoppingCriteria(
            sentinel_token_ids=tokenizer(
                "\nYou:",
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to("cuda"),
            starting_idx=tokenized_items.input_ids.shape[-1])
    ])

    logits = model.generate(stopping_criteria=stopping_criteria_list,
                            **tokenized_items,
                            **kwargs)
    output = tokenizer.decode(logits[0], skip_special_tokens=True)

    logger.debug("Before trimming, model output was: `%s`", output)

    # Trim out the input prompt from the generated output.
    if (idx := prompt.rfind(user_message)) != -1:
        trimmed_output = output[idx + len(user_message) - 1:].strip()
        logger.debug("After trimming, it became: `%s`", trimmed_output)

        return trimmed_output
    else:
        raise Exception(
            "Couldn't find user message in the model's output. What?")


def _build_bad_words_list_for(_model_name: str) -> t.List[str]:
    '''Builds a list of bad words for the given model.'''

    # NOTE(11b): This was implemented as a function because each model size
    # seems to have it quirks at the moment, but this is a rushed implementation
    # so I'm not handling that, hence the dumb return here.
    return ["Persona:", "Scenario:", "<START>"]


class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: torch.LongTensor,
                 starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor,
                 _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.sentinel_token_ids.shape[-1]:
                continue

            for window in trimmed_sample.unfold(
                    0, self.sentinel_token_ids.shape[-1], 1):
                if torch.all(torch.eq(self.sentinel_token_ids, window)):
                    return True
        return False


def _sanitize_string_for_use_in_a_regex(string: str) -> str:
    '''Sanitizes `string` so it can be used inside of a regexp.'''
    return BAD_CHARS_FOR_REGEX_REGEX.sub(r"\\\g<0>", string)


def parse_messages_from_str(string: str, names: t.List[str]) -> t.List[str]:
    '''
    Given a big string containing raw chat history, this function attempts to
    parse it out into a list where each item is an individual message.
    '''
    sanitized_names = [
        _sanitize_string_for_use_in_a_regex(name) for name in names
    ]

    speaker_regex = re.compile(rf"^({'|'.join(sanitized_names)}): ?",
                               re.MULTILINE)

    message_start_indexes = []
    for match in speaker_regex.finditer(string):
        message_start_indexes.append(match.start())

    # FIXME(11b): One of these returns is silently dropping the last message.
    if len(message_start_indexes) < 2:
        # Single message in the string.
        return [string.strip()]

    prev_start_idx = message_start_indexes[0]
    messages = []

    for start_idx in message_start_indexes[1:]:
        message = string[prev_start_idx:start_idx].strip()
        messages.append(message)
        prev_start_idx = start_idx

    return messages


def serialize_chat_history(history: t.List[str]) -> str:
    '''Given a structured chat history object, collapses it down to a string.'''
    return "\n".join(history)


def build_prompt_for(
    history: t.List[str],
    user_message: str,
    char_name: str,
    char_persona: t.Optional[str] = None,
    example_dialogue: t.Optional[str] = None,
    world_scenario: t.Optional[str] = None,
) -> str:
    '''Converts all the given stuff into a proper input prompt for the model.'''

    # If example dialogue is given, parse the history out from it and append
    # that at the beginning of the dialogue history.
    example_history = parse_messages_from_str(
        example_dialogue, ["You", char_name]) if example_dialogue else []
    concatenated_history = [*example_history, *history]

    # Construct the base turns with the info we already have.
    prompt_turns = [
        # TODO(11b): Shouldn't be here on the original 350M.
        "<START>",

        # TODO(11b): Arbitrary limit. See if it's possible to vary this
        # based on available context size and VRAM instead.
        *concatenated_history[-8:],
        f"You: {user_message}",
        f"{char_name}:",
    ]

    # If we have a scenario or the character has a persona definition, add those
    # to the beginning of the prompt.
    if world_scenario:
        prompt_turns.insert(
            0,
            f"Scenario: {world_scenario}",
        )

    if char_persona:
        prompt_turns.insert(
            0,
            f"{char_name}'s Persona: {char_persona}",
        )

    # Done!
    logger.debug("Constructed prompt is: `%s`", prompt_turns)
    prompt_str = "\n".join(prompt_turns)
    return prompt_str


def inference_fn(model: transformers.AutoModelForCausalLM, tokenizer: transformers.AutoTokenizer,
                 char_settings: t.List[str], history: t.List[str], user_input: str,
                     generation_settings=None,
                     ) -> str:

    # Brittle. Comes from the order defined in gradio_ui.py.
    if generation_settings is None:
        generation_settings = {
                "do_sample": True,
                "max_new_tokens": 256,
                "temperature": 0.5,
                "min_length": 100,
                "top_p": 0.9,
                "top_k": 50,
                "typical_p": 1.0,
                "repetition_penalty": 1.05,
            }
    [
            char_name,
            _user_name,
            char_persona,
            char_greeting,
            world_scenario,
            example_dialogue,
    ] = char_settings

        # If we're just starting the conversation and the character has a greeting
        # configured, return that instead. This is a workaround for the fact that
        # Gradio assumed that a chatbot cannot possibly start a conversation, so we
        # can't just have the greeting there automatically, it needs to be in
        # response to a user message.
    if len(history) == 0 and char_greeting is not None:
        return f"{char_name}: {char_greeting}"

    prompt = build_prompt_for(history=history,
                              user_message=user_input,
                              char_name=char_name,
                              char_persona=char_persona,
                              example_dialogue=example_dialogue,
                              world_scenario=world_scenario)

    if model and tokenizer:
        model_output = run_raw_inference(model, tokenizer, prompt,
                                             user_input, **generation_settings)
    else:
        raise Exception(
            "Not using local inference, but no Kobold instance URL was"
            " given. Nowhere to perform inference on.")

    generated_messages = parse_messages_from_str(model_output,
                                                     ["You", char_name])
    logger.debug("Parsed model response is: `%s`", generated_messages)
    bot_message = generated_messages[0]

    return bot_message
