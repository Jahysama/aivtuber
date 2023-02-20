#!/usr/bin/env python3
import sys
import threading
import queue
import time
from loguru import logger
from pathlib import Path
import contextlib

import pydantic

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
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
                                'Aimelia', #character name
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


@contextlib.contextmanager
def talking_face_generation():
    import os
    os.chdir('utils/MakeItTalk')
    import numpy
    import torchaudio
    import torch

    from src.approaches.train_audio2landmark import Audio2landmark_model
    from src.autovc.AutoVC_mel_Convertor_retrain_version import AutoVC_mel_Convertor
    from utils.talking_face_generation import opt_parser
    from utils.talking_face_generation import get_talking_head

    scale = numpy.float64(-0.01032258064516129)
    shift = numpy.array([-128.5, -82. ])
    shape_3d = numpy.load('shape3d.npy')
    landmarks = (shape_3d, scale, shift)

    c = AutoVC_mel_Convertor('examples')
    model = Audio2landmark_model(opt_parser, jpg_shape=shape_3d)

    def _talking_head(audio: numpy.ndarray):
        os.chdir('utils/MakeItTalk')
        torchaudio.save('generated.wav', torch.from_numpy(audio), 24000)
        video = get_talking_head('generated.wav', landmarks, c, model)
        os.chdir('../..')
        return video



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

    history = []
    model, tokenizer = build_model_and_tokenizer_for(settings.hf_model)

    def _generate(request: CompleteRequest):
        global history

        result = inference_fn(model=model,
                              tokenizer=tokenizer,
                              history=history[:4],
                              user_input=request.prompt,
                              generation_settings=None,
                              char_settings=settings.char_settings)
        result = result.replace(f"Aimelia:", f"**Aimelia:**") \
            .replace("<USER>", "Chat").replace('\n', ' ')

        result = result.split('*')
        result = [v for i, v in enumerate(result) if i % 2 == 0]
        result = " ".join(result)

        history.append(f"You: {request.prompt}")
        history.append(result)

        return result.replace('Aimelia: ', '')

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
                    video = talking_face_generation_fn(audio)
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
                    response_queue.put({'response': response, 'emotion': emotion, 'audio': audio, 'video': video})
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
    _enqueue(CompleteRequest(prompt="hello"))


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
