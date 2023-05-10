#github test
from threading import Thread
import time
import os
import sys

import numpy as np
#import torch

import argparse
#import pyvirtualcam
import time
import requests
import sounddevice as sd
#import util.utils as util
from scipy.signal import savgol_filter
import cv2
from PIL import Image
import websocket
import time
import socket


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



#cam = pyvirtualcam.Camera(width=256, height=256, fps=90, device='/dev/video3')
#cap = cv2.VideoCapture('/home/keeper/Videos/anya_idle.mp4')
#frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

#fc = 0
#ret = True

#while (fc < frameCount  and ret):
#    ret, buf[fc] = cap.read()
#    fc += 1
#cap.release()

thread_running = True
pause = False
screenshots = []


def my_forever_while():
    global thread_running
    global pause
    global screenshots

    while thread_running:
            for i in range(len(buf)):
                if not pause:
                    frame = buf[i] * 255.0
                    cam.send(cv2.bitwise_not(frame.astype(np.uint8)[..., ::-1]))
                    #cam.sleep_until_next_frame()
                    time.sleep(1 / 60)


def take_screenshot():
    global thread_running
    global pause
    global screenshots
    if len(screenshots) > 30:
        screenshots.pop(0)
    os.system("import -silent -window root vision.png")
    image = Image.open('vision.png').convert(mode="RGB")
    screenshots.append(np.array(image).tolist())
    time.sleep(2)


def on_message(ws, message):
    sock.sendto(message, ("127.0.0.1", 11574))


def send_landmarks():
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8080/ws",
                              on_message = on_message)
    ws.run_forever()


def take_input():
    global thread_running
    global pause

    while thread_running:
        user_input = input('Input text: ')
        resp = requests.post('http://localhost:8000/complete',
                             json={'prompt': user_input, 'api_key': 'superkey', 'screen': screenshots}).json()['response']
        #video = make_video(np.array(resp['video']), resp['audio_emb'])
        voice = np.array(resp['audio'])

        #vid = []
        print(resp['response'])
        #for i in range(len(video)):
        #    frame = video[i] * 255.0
        #    vid.append(frame.astype(np.uint8)[..., ::-1])
        #pause = True
        sd.play(voice, 24000)
        #for frame in vid:
        #    cam.send(frame)
        #    cam.sleep_until_next_frame()
            #time.sleep(1 / 375)
        #pause = False



if __name__ == '__main__':
    t1 = Thread(target=send_landmarks)
    t2 = Thread(target=take_input)
    t3 = Thread(target=take_screenshot)

    t1.start()
    t2.start()
    t3.start()

    t2.join()  # interpreter will wait until your process get completed or terminated
    t1.join()

    thread_running = False
