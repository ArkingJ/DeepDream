import numpy as np
import PIL.Image
from io import BytesIO
import IPython.display as ipd
from IPython.display import display
import moviepy.editor as mpy
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm.notebook import tqdm
import subprocess
import pandas as pd
import notebook

t_input =r"C:\Users\OWEN\PycharmProjects\pythonProject5\video_2024-05-24_13-53-08.mp4"


def deepdream(graph, t_input, t_preprocessed, layers, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        sess = tf.InteractiveSession(graph=graph)
        for y in range(0, max(h - sz // 2, sz), sz):
            for x in range(0, max(w - sz // 2, sz), sz):
                sub = img_shift[y:y + sz, x:x + sz]
                g = sess.run(t_grad, {t_input: sub})
                grad[y:y + sz, x:x + sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    # Split the image into a number of octaves
    img = t_input
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # Generate details octave by octave
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))

    # Output the final image
    return img


def resize(img, size):
    img = np.float32(img)
    return np.array(PIL.Image.fromarray(np.uint8(img)).resize(size, PIL.Image.LANCZOS))


def save_image(img, filename):
    PIL.Image.fromarray(np.uint8(np.clip(img, 0, 255))).save(filename)


def preprocess_image(image_path):
    img = PIL.Image.open(image_path)
    img = np.float32(img)
    return img


def render_deep_dream(graph, t_input, t_preprocessed, layers, image_path, iterations, save_path):
    img = preprocess_image(image_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        t_grad = tf.gradients(tf.reduce_mean(tf.square(layers)), t_preprocessed)[0]
        for i in range(iterations):
            img = deepdream(graph, img, t_preprocessed, layers)
        save_image(img, save_path)


def process_video(video_path, output_path, graph, t_input, t_preprocessed, layers, frame_rate=30):
    video = mpy.VideoFileClip(video_path)
    dream_video = video.fl_image(lambda frame: np.clip(deepdream(graph, frame, t_preprocessed, layers), 0, 255))
    dream_video.write_videofile(output_path, fps=frame_rate)


ipd.Video(t_input)


