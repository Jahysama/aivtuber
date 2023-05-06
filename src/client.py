#github test
from threading import Thread
import time
import os
import sys
sys.path.append("/home/keeper/Code/aivtuber-github/src/utils/MakeItTalk")
os.chdir('utils/MakeItTalk')

import numpy as np
import torch
import argparse
#import pyvirtualcam
import time
import requests
import sounddevice as sd
import util.utils as util
from scipy.signal import savgol_filter
from src.approaches.train_image_translation import Image_translation_block
import cv2
from PIL import Image
import websocket
import time
import socket


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


default_head_name = 'anya'  # the image name (with no .jpg) to animate
ADD_NAIVE_EYE = True  # whether add naive eye blink
CLOSE_INPUT_FACE_MOUTH = False  # if your image has an opened mouth, put this as True, else False
AMP_LIP_SHAPE_X = 3.5  # amplify the lip motion in horizontal direction
AMP_LIP_SHAPE_Y = 4.1  # amplify the lip motion in vertical direction
AMP_HEAD_POSE_MOTION = 0.95  # amplify the head pose motion (usually smaller than 1.0, put it to 0. for a static head pose)
scale = np.float64(-0.00963855421686747)
shift = np.array([-122., -105.])

parser = argparse.ArgumentParser()
parser.add_argument('--jpg', type=str, default='{}.jpg'.format(default_head_name))
parser.add_argument('--close_input_face_mouth', default=CLOSE_INPUT_FACE_MOUTH, action='store_true')

parser.add_argument('--load_AUTOVC_name', type=str, default='examples/ckpt/ckpt_autovc.pth')
parser.add_argument('--load_a2l_G_name', type=str, default='examples/ckpt/ckpt_speaker_branch.pth')
parser.add_argument('--load_a2l_C_name', type=str,
                    default='examples/ckpt/ckpt_content_branch.pth')  # ckpt_audio2landmark_c.pth')
parser.add_argument('--load_G_name', type=str,
                    default='examples/ckpt/ckpt_116_i2i_comb.pth')  # ckpt_image2image.pth') #ckpt_i2i_finetune_150.pth') #c

parser.add_argument('--amp_lip_x', type=float, default=AMP_LIP_SHAPE_X)
parser.add_argument('--amp_lip_y', type=float, default=AMP_LIP_SHAPE_Y)
parser.add_argument('--amp_pos', type=float, default=AMP_HEAD_POSE_MOTION)
parser.add_argument('--reuse_train_emb_list', type=str, nargs='+',
                    default=[])  # ['iWeklsXc0H8']) #['45hn7-LXDX8']) #['E_kmpT-EfOg']) #'iWeklsXc0H8', '29k8RtSUjE0', '45hn7-LXDX8',
parser.add_argument('--add_audio_in', default=False, action='store_true')
parser.add_argument('--comb_fan_awing', default=False, action='store_true')
parser.add_argument('--output_folder', type=str, default='examples')

parser.add_argument('--test_end2end', default=False, action='store_true')
parser.add_argument('--dump_dir', type=str, default='', help='')
parser.add_argument('--pos_dim', default=7, type=int)
parser.add_argument('--use_prior_net', default=True, action='store_true')
parser.add_argument('--transformer_d_model', default=32, type=int)
parser.add_argument('--transformer_N', default=2, type=int)
parser.add_argument('--transformer_heads', default=2, type=int)
parser.add_argument('--spk_emb_enc_size', default=16, type=int)
parser.add_argument('--init_content_encoder', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-6, help='weight decay')
parser.add_argument('--write', default=False, action='store_true')
parser.add_argument('--segment_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--emb_coef', default=3.0, type=float)
parser.add_argument('--lambda_laplacian_smooth_loss', default=1.0, type=float)
parser.add_argument('--use_11spk_only', default=False, action='store_true')
parser.add_argument('-f')

opt_parser = parser.parse_args()


def make_video(fls, audio_emb):

    img = cv2.imread('examples/' + opt_parser.jpg)
    for i in range(0,len(fls)):
        fl = np.array(audio_emb[0]).reshape((-1, 68, 3))
        fl[:, :, 0:2] = -fl[:, :, 0:2]
        fl[:, :, 0:2] = fl[:, :, 0:2] / scale - shift

        if (ADD_NAIVE_EYE):
            fl = util.add_naive_eye(fl)

        # additional smooth
        fl = fl.reshape((-1, 204))
        fl[:, :48 * 3] = savgol_filter(fl[:, :48 * 3], 15, 3, axis=0)
        fl[:, 48*3:] = savgol_filter(fl[:, 48*3:], 5, 3, axis=0)
        fl = fl.reshape((-1, 68, 3))

        ''' STEP 6: Imag2image translation '''
        model = Image_translation_block(opt_parser, single_test=True)
        with torch.no_grad():
            video = model.single_test(jpg=img, fls=fl, filename=fls[i], prefix=opt_parser.jpg.split('.')[0])
            print('finish image2image gen')

    return video


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
        sd.play(voice.T, 24000)
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
