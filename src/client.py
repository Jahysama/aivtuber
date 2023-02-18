import numpy
import pyvirtualcam
import time
import sounddevice as sd

if __name__ == "__main__":
    cam = pyvirtualcam.Camera(width=256, height=256, fps=60, device='/dev/video4')
    vid = numpy.load('/home/keeper/Downloads/video.npy')
    video = []
    for i in range(500):
        video.append(vid)
    res = numpy.concatenate(video, axis=0)
    print(res.shape)
    voice = numpy.load('/home/keeper/Documents/test_voice.npy')
    while True:
        sd.play(voice.T, 24000)
        for i in range(res.shape[0]):
            frame = res[i] * 255.0
            cam.send(frame.astype(numpy.uint8)[..., ::-1])
            cam.sleep_until_next_frame()
            print(i)
