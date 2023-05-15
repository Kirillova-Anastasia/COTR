'''
COTR demo for a single image pair
'''
import argparse
import os
import time
import datetime

import cv2
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt

from COTR.utils import utils, debug_utils
from COTR.models import build_model
from COTR.options.options import *
from COTR.options.options_utils import *
from COTR.inference.inference_helper import triangulate_corr
from COTR.inference.sparse_engine import SparseEngine

utils.fix_randomness(0)
torch.set_grad_enabled(False)


def main(opt):
    model = build_model(opt)
    device = 'cuda' if opt.use_cuda else 'cpu'
    model = model.to(device)
    weights = torch.load(opt.load_weights_path, map_location='cpu')['model_state_dict']
    utils.safe_load_weights(model, weights)
    model = model.eval()

    with open('Pitts30k_database_paths.txt', 'r') as f:
        database_paths = [line.split()[1] for line in f]
    with open('Pitts30k_queries_paths.txt', 'r') as f:
        queries_paths = [line.split()[1] for line in f]
    with open('Pitts30k_positives.txt', 'r') as f:
        positives = {int(line.split()[0]): int(line.split()[12]) for line in f}

    for opt.max_corrs in [20, 50, 100]:
        for query_index, query_path in enumerate(queries_paths):
            opt.exp_name = f'pitts_{query_index}_default_{opt.max_corrs}corrs'
            with open('log.txt', 'a') as f:
                f.write(f'Start {opt.exp_name} at {datetime.datetime.now()}\n')

            database_index = positives[query_index]
            database_path = database_paths[database_index]

            # img_a = imageio.imread(opt.first_path, pilmode='RGB')
            # img_b = imageio.imread(opt.second_path, pilmode='RGB')
            img_a = imageio.imread(query_path, pilmode='RGB')
            img_b = imageio.imread(database_path, pilmode='RGB')

            engine = SparseEngine(model, 32, mode='tile')

            if opt.just_plot:
                with open(f'saved_corrs/{opt.exp_name}.npy', 'rb') as f:
                    corrs = np.load(f)
            else:
                t0 = time.time()
                corrs = engine.cotr_corr_multiscale_with_cycle_consistency(img_a, img_b, np.linspace(0.5, 0.0625, 4), 1, max_corrs=opt.max_corrs, queries_a=None)
                t1 = time.time()
                print(f'spent {t1-t0} seconds for {opt.max_corrs} correspondences.')
                with open(f'saved_corrs/{opt.exp_name}.npy', 'wb') as f:
                    np.save(f, corrs)

            utils.save_visualized_corrs(img_a, img_b, corrs, exp_name=opt.exp_name)
            with open('log.txt', 'a') as f:
                f.write(f'Finish {opt.exp_name} at {datetime.datetime.now()}\n\n')
            # utils.visualize_corrs(img_a, img_b, corrs, exp_name=opt.exp_name)
            # dense = triangulate_corr(corrs, img_a.shape, img_b.shape)
            # warped = cv2.remap(img_b, dense[..., 0].astype(np.float32), dense[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # plt.imshow(warped / 255 * 0.5 + img_a / 255 * 0.5)
            # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_COTR_arguments(parser)
    parser.add_argument('--out_dir', type=str, default=general_config['out'], help='out directory')
    parser.add_argument('--load_weights', type=str, default=None, help='load a pretrained set of weights, you need to provide the model id')
    parser.add_argument('--max_corrs', type=int, default=100, help='number of correspondences')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='use cuda')
    
    # parser.add_argument('--exp_name', type=str, required=True, help='Name of experiment, equal name of result file')
    parser.add_argument('--just_plot', action='store_true', default=False, help='If just read and plot corrs or calculate them first')
    # parser.add_argument('--first_path', type=str, required=True, help='Path to first image')
    # parser.add_argument('--second_path', type=str, required=True, help='Path to second image')
    

    opt = parser.parse_args()
    opt.command = ' '.join(sys.argv)

    layer_2_channels = {'layer1': 256,
                        'layer2': 512,
                        'layer3': 1024,
                        'layer4': 2048, }
    opt.dim_feedforward = layer_2_channels[opt.layer]
    if opt.load_weights:
        opt.load_weights_path = os.path.join(opt.out_dir, opt.load_weights, 'checkpoint.pth.tar')
    print_opt(opt)
    main(opt)
    
