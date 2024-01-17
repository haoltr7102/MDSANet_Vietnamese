# -*- coding: utf-8 -*-
import sys; sys.path.append('./scene_graph_benchmark/scene_graph_benchmark')
from scene_graph_benchmark.scene_parser import SceneParser
from scene_graph_benchmark.AttrRCNN import AttrRCNN
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.config import cfg
from scene_graph_benchmark.config import sg_cfg
from maskrcnn_benchmark.data.datasets.utils.load_files import \
    config_dataset_file
from maskrcnn_benchmark.data.datasets.utils.load_files import load_labelmap_file
from maskrcnn_benchmark.utils.miscellaneous import mkdir

import torch
import clip
from PIL import Image

import os
import glob
import cv2
import torch
from PIL import Image
import numpy as np
#import tqdm
import json
import h5py
import torch.nn as nn
from tqdm import tqdm

import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.MDSANet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, TransformerEnsemble
import torch
import argparse
import pickle
import itertools
import matplotlib.pyplot as plt

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)



def extract_clip_feature(image, device):
    # image = preprocess(image).unsqueeze(0).to(device)
    # image = image.type(torch.HalfTensor).to(device)
    # with torch.no_grad():
    #     x = clip_model.visual.conv1(image)  # shape = [*, width, grid, grid]
    image = preprocess(image).unsqueeze(0).to(device)
    image = image.type(torch.FloatTensor).to(device)  
    with torch.no_grad():
        x = clip_model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + clip_model.visual.positional_embedding.to(x.dtype)
        x = clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.visual.ln_post(x[:, 0, :])
    return x.view(x.size(0), 64, -1)


#Setting configuration
cfg.set_new_allowed(True)
cfg.merge_from_other_cfg(sg_cfg)
cfg.set_new_allowed(False)
#Configuring VinVl
cfg.merge_from_file('./scene_graph_benchmark/sgg_configs/vgattr/vinvl_x152c4.yaml')
argument_list = [
                 'MODEL.WEIGHT', './../model/vinvl_vg_x152c4.pth',
                 'MODEL.ROI_HEADS.NMS_FILTER', 1,
                 'MODEL.ROI_HEADS.SCORE_THRESH', 0.2,
                 'TEST.IGNORE_BOX_REGRESSION', False,
                 'MODEL.ATTRIBUTE_ON', True,
                 'MODEL.DEVICE', 'cuda:0',
                 'TEST.OUTPUT_FEATURE', True,
]

cfg.merge_from_list(argument_list)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR

model = AttrRCNN(cfg)
model.to(cfg.MODEL.DEVICE)
model.eval()

transforms = build_transforms(cfg, is_train=False)
checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
checkpointer.load(cfg.MODEL.WEIGHT)

def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img

import numpy as np

# Extract features

# dict_dataset_img = {
#     'img': './../dataset/test',
# }

import torch
torch.cuda.empty_cache()

#for dataset_img in dict_dataset_img:
img_path = './../dataset/test/vietnam.jpg'
print('Extracting image')
save_dir = './../model/feature'
save_hdf5 = h5py.File(os.path.join(save_dir, 'demo_img_clip_features.hdf5'), 'w')

#for img_path in tqdm(img_paths):

dict_img = {}
filename = os.path.basename(img_path)
try:
    image = cv2.imread(img_path)
    img_input = cv2Img_to_Image(image)
except:
    image = Image.open(img_path)
    img_input = np.array(image)
    if img_input.shape[-1] < 3:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
    img_input = cv2Img_to_Image(img_input)

img_fit_to_get_clip = img_input
img_input, _ = transforms(img_input, target=None)
img_input = img_input.to(cfg.MODEL.DEVICE)
raw_height, raw_width = img_input.shape[-2:]

with torch.no_grad():
    prediction = model(img_input.type(torch.FloatTensor))[0].to('cpu')

prediction = prediction.resize((raw_width, raw_height))
det_dict = {key : prediction.get_field(key) for key in prediction.fields()}
box_features = det_dict['box_features']

# Get boxes
boxes_all = torch.zeros(box_features.size(0), 4)
max_indexes = np.argmax(det_dict['scores_all'], axis=-1)
for idx, (max_idx, box) in enumerate(zip(max_indexes, det_dict['boxes_all'])):
    boxes_all[idx] = box[max_idx]

# Get CLIP features
clip_grid_features = extract_clip_feature(img_fit_to_get_clip, 'cpu') # cuda:0

save_hdf5.create_dataset(filename.split('.')[0] + '_features', data=box_features.cpu().detach().numpy())
#save_hdf5.create_dataset(filename.split('.')[0] + '_boxes', data=boxes_all.cpu().detach().numpy())
save_hdf5.create_dataset(filename.split('.')[0] + '_grids', data=clip_grid_features.cpu().detach().numpy())

save_hdf5.close()


# Generate caption

def visualize_result(image_dir, caption, output_path):
    image_ids = os.path.basename(image_dir)
    image_path = os.path.join(image_dir)
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error when read img {image_ids}: {e}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the figure to the output folder
    output_path = os.path.join(output_folder, f'{image_ids}')

    try:
        plt.imshow(np.array(image))
        plt.axis('off')

        # Add the caption below the image
        plt.text(0.5, -0.1, "Caption: " + caption, ha='center', va='center', transform=plt.gca().transAxes)
        plt.savefig(output_path)
        plt.show()

    except Exception as e:
        print(f"Error when save img {image_ids}: {e}")
    finally:
        plt.close()


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='MDSANet')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--features_path', type=str, default='/home/data/coco_grid_feats2.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./annotation')
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    print('MDSANet Evaluation')

    features_path = './../model/feature/demo_img_clip_features.hdf5'

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    text_field.vocab = pickle.load(open('./vocab_uit_viic.pkl', 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 130, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    model_path = './../model/MDSANet_best_CLIP_UIT_ViIC.pth'
    data = torch.load(model_path)
    model.load_state_dict(data['state_dict'])

    # image_ids = [os.path.basename(i) for i in glob.glob(os.path.join(img_path, '*'))]   # vs code
    image_ids = [os.path.basename(img_path)]

    # image_ids = [i.split('/')[-1] for i in \
    #         glob.glob(os.path.join(img_path, '*'))]   # gg colab
    print(image_ids)
    
    f = h5py.File(features_path, 'r')

    results = []
    max_detections = 49

    output_folder = './../output_img_demo'

    for image_name in tqdm(image_ids):
        image = f['%s_features' % image_name.split('.')[0]][()]
        delta = max_detections - image.shape[0]
        if delta > 0:
            image = np.concatenate([image, np.zeros((delta, image.shape[1]))], axis=0)
        elif delta < 0:
            image = image[:max_detections]
        torch_image = torch.from_numpy(np.array([image]).astype(np.float32)).to(device)
        with torch.no_grad():
            out, _ = model.beam_search(torch_image, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
        caps_gen = text_field.decode(out, join_words=False)
        gen_i = ' '.join([k for k, g in itertools.groupby(caps_gen[0])])
        gen_i = gen_i.strip().replace('_', ' ')
        results.append({"id": image_name, "captions": gen_i})
        visualize_result(img_path, gen_i, output_folder)  