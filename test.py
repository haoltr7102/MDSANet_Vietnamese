import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from models.MDSANet import Transformer, TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention, TransformerEnsemble
import torch
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import h5py
import glob
import itertools
import json
import matplotlib.pyplot as plt
from PIL import Image

def predict_captions(model, dataloader, text_field):
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        for it, (images, caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                print(gen_i)
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores

def visualize_result(image_dir, image_name, caption, output_folder):
    image_path = os.path.join(image_dir, image_name)

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error when read img {image_name}: {e}")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Save the figure to the output folder
    output_path = os.path.join(output_folder, f'{image_name}')

    try:
        plt.imshow(np.array(image))
        plt.axis('off')

        # Add the caption below the image
        plt.text(0.5, -0.1, "Caption: " + caption, ha='center', va='center', transform=plt.gca().transAxes)

        plt.savefig(output_path)
    except Exception as e:
        print(f"Error when save img {image_name}: {e}")
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

    # Pipeline for image regions
    image_field = ImageDetectionsField(detections_path=args.features_path, max_detections=49, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Create the dataset
    text_field.vocab = pickle.load(open('/content/image-captioning-MDSANet/vocab.pkl', 'rb'))

    # Model and dataloaders
    encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention)
    decoder = TransformerDecoderLayer(len(text_field.vocab), 130, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)

    model_path = '/content/drive/MyDrive/saved_transformer_models/UIT_OpenViIC/MDSANet_best.pth'
    data = torch.load(model_path)
    model.load_state_dict(data['state_dict'])


    img_path = '/content/drive/MyDrive/datasets/OpenViIC/test' #//public/ private
    features_path = '/content/drive/MyDrive/OpenViIC_test_vinvl_region_features.hdf5'
    image_ids = [i.split('/')[-1] for i in \
            glob.glob(os.path.join(img_path, '*'))]
    print(image_ids)
    
    f = h5py.File(features_path, 'r')

    results = []
    max_detections = 49

    output_folder = '/content/drive/MyDrive/OpenViIC/Output_OpenViIC_img'

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
        visualize_result(img_path, image_name, gen_i, output_folder)
    

    # path_sample_submission = '/content/vietcap4h-public-test/sample_submission.json'
    # with open(path_sample_submission, 'r') as f:
    #     sample_submission = json.load(f)

    # for item in sample_submission:
    #     for result in results:
    #         if item['id'] == result['id']:
    #             item['captions'] = result['captions'].split(' ')[0].capitalize() + ' ' + ' '.join(result['captions'].split(' ')[1:]) + '.'
    #             continue

    # with open('results.json', 'w') as f:
    #     json.dump(sample_submission, f, indent=4, ensure_ascii=False)