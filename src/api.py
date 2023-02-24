#!/usr/bin/env python3
import threading
import queue
import time
from loguru import logger
from pathlib import Path
import contextlib

import pydantic

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = FastAPI()

origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )


class Settings(pydantic.BaseSettings):
    queue_size: int = 1024
    log_file: str = "../server.log"
    api_keys_file: str = '../valid_api_keys.txt'
    hf_model: str = 'PygmalionAI/pygmalion-6b'
    text_classification: str = 'j-hartmann/emotion-english-distilroberta-base'
    voice: str = 'emma'
    voice_quality: str = 'ultra_fast'
    char_settings: list[str] = [
                                'Noisy chan', #character name
                                'Chat', #user name
                                'A very cool twitch streamer and gamer, also AI powered by GPT-J. Streamer is very tolerant and does not say any slurs.',
                                'Hello',
                                'Streamer interacts with live chat',
                                ''
                               ]


settings = Settings()

def _check_api_key(key):
    key = key.strip()
    for line in Path(settings.api_keys_file).open():
        if not line:
            continue
        valid_key = line.split()[0]
        if key == valid_key:
            break
    else:
        return False
    return True

request_queue = queue.Queue(maxsize=settings.queue_size)

history = []

@contextlib.contextmanager
def talking_face_generation():
    import os
    os.chdir('utils/MakeItTalk')
    import numpy
    import torchaudio
    import torch

    from utils.talking_face_generation import get_talking_head

    scale = numpy.float64(-0.00963855421686747)
    shift = numpy.array([-122., -105.])
    shape_3d = numpy.load('../../shape3d.npy')
    landmarks = (shape_3d, scale, shift)


    os.chdir('../..')

    def _talking_head(audio: numpy.ndarray):
        os.chdir('utils/MakeItTalk')
        #os.remove(f'examples/generated.wav')
        torchaudio.save(f'examples/generated.wav', torch.from_numpy(audio), 24000)
        video = get_talking_head(f'generated.wav', landmarks)
        video = get_talking_head(f'generated.wav', landmarks)
        audio_embs = []
        for i in range(0, len(video)):
            audio_emb = numpy.loadtxt(f'examples/{video[i]}').tolist()
            audio_embs.append(audio_emb)
        os.chdir('../..')
        return video, audio_embs



    yield _talking_head



@contextlib.contextmanager
def audio_generation():
    from tortoise.api import TextToSpeech
    from utils.audio_generation import get_voice

    tts = TextToSpeech()

    def _voice(text: str):
        audio = get_voice(text, tts, settings.voice, settings.voice_quality)
        return audio

    yield _voice


@contextlib.contextmanager
def emo_detection():
    from utils.emotion_detection import get_emotion
    from transformers import pipeline

    classifier = pipeline("text-classification", model=settings.text_classification,
                          return_all_scores=True)

    def _emotion(text: str):
        emo = get_emotion(text, classifier)
        return emo

    yield _emotion


@contextlib.contextmanager
def hf_generation():
    from utils.text_generation import build_model_and_tokenizer_for
    from utils.text_generation import inference_fn

    from transformers import pipeline
    from langchain import HuggingFacePipeline
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    
    from llama_index import LangchainEmbedding
    from llama_index import LLMPredictor, PromptHelper
    from llama_index import SimpleDirectoryReader
    from scripts import GPTSimpleVectorIndexContext

    from os import path

    model, tokenizer = build_model_and_tokenizer_for(settings.hf_model)
    
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
    )

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    llm_predictor = LLMPredictor(llm=HuggingFacePipeline(pipeline=pipe))
    max_input_size = 512
    num_output = 256
    max_chunk_overlap = 256
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    if path.exists('../index.json'):
        index = GPTSimpleVectorIndexContext.load_from_disk('../index.json', llm_predictor=llm_predictor,
                                                    embed_model=embed_model, prompt_helper=prompt_helper)
    else:
        documents = SimpleDirectoryReader('../data').load_data()
        index = GPTSimpleVectorIndexContext(documents, llm_predictor=llm_predictor, embed_model=embed_model,
                                     prompt_helper=prompt_helper)
        index.save_to_disk('../index.json')

    def _generate(request: CompleteRequest):
        global history

        query = index.query(request.prompt)
        similarities = [similarity for _, similarity in query]
        text = [node.get_text() for node, _ in query]
        max_sim_index = similarities.index(max(similarities))
        char_settings = settings.char_settings
        if max(similarities) > 0.1:
            char_settings[2] = char_settings[2] + '\n'\
                                    + text[max_sim_index]

        result = inference_fn(model=model,
                              tokenizer=tokenizer,
                              history=history[:4],
                              user_input=request.prompt,
                              generation_settings=None,
                              char_settings=char_settings)
        result = result.replace(f"Noisy chan:", f"**Noisy chan:**") \
            .replace("<USER>", "Chat").replace('\n', ' ')

        result = result.split('*')
        result = [v for i, v in enumerate(result) if i % 2 == 0]
        result = " ".join(result)

        history.append(f"You: {request.prompt}")
        history.append(result)

        return result.replace('Noisy chan: ', '')[3:]

    yield _generate


def worker():
    generation = hf_generation
    with generation() as generate_fn, emo_detection() as emo_detection_fn, \
            audio_generation() as audio_generation_fn, \
            talking_face_generation() as talking_face_generation_fn:
        with open(settings.log_file, "a") as logf:
            while True:
                response_queue = None
                try:
                    start_time = time.time()
                    (request, response_queue) = request_queue.get()
                    logger.info(f"getting request took {time.time() - start_time}")
                    start_time = time.time()
                    response = generate_fn(request)
                    emotion = emo_detection_fn(response)
                    audio = audio_generation_fn(response)
                    video, audio_emb = talking_face_generation_fn(audio)
                    logger.info(f"generate took {time.time() - start_time}, response length: {len(response)}")
                    start_time = time.time()

                    logf.write(f"##### {request.api_key} ##### {time.time()} #####\n")
                    logf.write("###\n")
                    logf.write(f"{request.prompt}\n")
                    logf.write("#####\n")
                    logf.write(f"{response}\n\n")
                    logf.flush()

                    logger.info(f"writing log took {time.time() - start_time}")
                    start_time = time.time()
                    response_queue.put({'response': response, 'emotion': emotion, 'audio': audio.tolist(), 'video': video, 'audio_emb': audio_emb})
                    logger.info(f"putting response took {time.time() - start_time}")
                except KeyboardInterrupt:
                    logger.info(f"Got KeyboardInterrupt... quitting!")
                    raise
                except Exception:
                    logger.exception(f"Got exception, will continue")
                    if response_queue is not None:
                        response_queue.put("")



@app.get("/")
async def main():
    return {"response": "Hello, world!"}

class CompleteRequest(pydantic.BaseModel):
    prompt: pydantic.constr(min_length=0, max_length=2**14)
    api_key: pydantic.constr(min_length=1, max_length=128) = "x"*9

def _enqueue(request: CompleteRequest):
    response_queue = queue.Queue()
    request_queue.put((request, response_queue))
    response = response_queue.get()
    return response


@app.on_event("startup")
def startup():
    threading.Thread(
            target=worker,
            daemon=True,
            ).start()


@app.post("/complete")
def complete(request: CompleteRequest):
    logger.info(f"Received request from key {request.api_key}. Queue size is {request_queue.qsize()}")
    if request_queue.full():
        logger.warning("Request queue full.")
        raise ValueError("Request queue full.")
    if not _check_api_key(request.api_key):
        logger.warning(f"api key not valid: {request.api_key}, discarding...")
        raise ValueError("Invalid API key")
    response = _enqueue(request)
    return {"response": response}
