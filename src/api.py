#!/usr/bin/env python3
import threading
import queue
import time
from loguru import logger
from pathlib import Path
import contextlib

import pydantic

import cv2
import numpy

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
                                'Your name is Noisy chan', #character name
                                'You are talking with a Chat', #user name
                                'You follow a personality of a very cool twitch streamer and gamer, also AI powered by Llama. You are is very tolerant and does not say any slurs.'
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

pause_idle_animation = False


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

    def _talking_head():
        os.chdir('utils/MakeItTalk')
        #os.remove(f'examples/generated.wav')
        video = get_talking_head(f'generated.wav', landmarks)
        os.chdir('../..')
        return video

    yield _talking_head



def prepare_virtual_camera():
    import pyvirtualcam
    import cv2
    import numpy

    cam = pyvirtualcam.Camera(width=256, height=256, fps=62.5, device='/dev/video0')
    cap = cv2.VideoCapture('idle.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = numpy.empty((frameCount, frameHeight, frameWidth, 3), numpy.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()

    return cam, buf


cam, buf = prepare_virtual_camera()


@contextlib.contextmanager
def audio_generation():
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    preload_models()

    def _voice(text: str):
        audio_array = generate_audio(text)

        # save audio to disk
        write_wav("utils/MakeItTalk/examples/", SAMPLE_RATE, audio_array)

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
    import torch

    from utils.text_generation import build_model_and_tokenizer_for
    from utils.text_generation import inference_fn

    from transformers import pipeline
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
    from langchain import HuggingFacePipeline
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    
    from llama_index import LangchainEmbedding
    from llama_index import LLMPredictor, PromptHelper
    from llama_index import SimpleDirectoryReader
    from scripts.GPTSimpleVectorIndexContext import GPTSimpleVectorIndexContext

    from PIL import Image
    from os import path
    import numpy as np

    model, tokenizer = build_model_and_tokenizer_for()
    image2text = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base",  torch_dtype=torch.float16).to("cuda")
    processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-base")

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256
    )

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())
    llm_predictor = LLMPredictor(llm=HuggingFacePipeline(pipeline=pipe))
    max_input_size = 270
    num_output = 128
    max_chunk_overlap = 100
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
        if max(similarities) > 0.4:
            logger.info(f"Context: {text[max_sim_index]}")
            char_settings[2] = char_settings[2] +\
                                '\nConsider and follow this information:\n' +\
                                text[max_sim_index]

        view_over_time = []
        for image in request.screen:
            raw_image = Image.fromarray(np.array(image).astype('uint8'))
            inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
            out = image2text.generate(**inputs)
            view_over_time.append(processor.decode(out[0], skip_special_tokens=True))
        view = "\n".join(view_over_time)
        logger.info(f'View: {view}')
        char_settings[2] = char_settings[2] + \
                           '\nStreamer sees what is happening on the screen over time:\n' + \
                           view
        char_settings[2] = char_settings[2] + '\nHere is a history of a current conversation:\n' + \
                            '\n'.join(history[:3])

        char_settings = "\n".join(char_settings)

        result = inference_fn(model=model,
                              tokenizer=tokenizer,
                              user_input=request.prompt,
                              generation_settings=None,
                              char_settings=char_settings)

        result_final = 'Noisy chan: ' + result
        result = result.replace("Chat", "User")

        history.append(f"Chat: {request.prompt}")
        history.append(result_final)

        return result

    yield _generate


def worker():
    global pause_idle_animation
    global cam
    global buf
    import torch

    torch.cuda.empty_cache()
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
                    audio_generation_fn(response)
                    video = talking_face_generation_fn()
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
                    response_queue.put({'response': response, 'emotion': emotion, 'audio': audio.tolist()})
                    vid = []
                    for i in range(len(video)):
                        frame = video[i] * 255.0
                        vid.append(frame.astype(numpy.uint8)[..., ::-1])
                    pause_idle_animation = True
                    for frame in vid:
                        cam.send(frame)
                        cam.sleep_until_next_frame()
                    logger.info(f"putting response took {time.time() - start_time}")
                    pause_idle_animation = False
                except KeyboardInterrupt:
                    logger.info(f"Got KeyboardInterrupt... quitting!")
                    raise
                except Exception:
                    logger.exception(f"Got exception, will continue")
                    if response_queue is not None:
                        response_queue.put("")


def stream_video():
    global pause_idle_animation
    global cam
    global buf
    while True:
        for i in range(len(buf)):
            if not pause_idle_animation:
                frame = buf[i] * 255.0
                cam.send(cv2.bitwise_not(frame.astype(numpy.uint8)[..., ::-1]))
                cam.sleep_until_next_frame()
                # time.sleep(1 / 60)


@app.get("/")
async def main():
    return {"response": "Hello, world!"}

class CompleteRequest(pydantic.BaseModel):
    prompt: pydantic.constr(min_length=0, max_length=2**14)
    api_key: pydantic.constr(min_length=1, max_length=128) = "x"*9
    screen: list[list]

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
