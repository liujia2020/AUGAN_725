import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time
from options.test_options import TestOptions
from models import create_model
from data_process import load_dataset, test_image
from torch.utils.data import DataLoader,Dataset,TensorDataset
from utils.metrics import image_evaluation
from cubdl.example_picmus_torch import load_datasets,create_network,mk_img,dispaly_img

if __name__ == '__main__':
    # Initial setting and correponding parameters
    plane_wave_data = load_datasets("simulation", "resolution_distorsion", "iq")
    das, iqdata, xlims, zlims = create_network(plane_wave_data, [1])

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    img_eva = image_evaluation()

    test_type = 2 # Tese different types of dataset
    img_dataset = load_dataset(opt, opt.phase, test_type)
    dataset_len = img_dataset.len
    train_loader = DataLoader(dataset=img_dataset, num_workers=0, batch_size=1, shuffle=False)

    train_bar = tqdm(train_loader)
    i = 1
    # Testing process
    for data, target5 in train_bar:
        n1, n2, n3 = data.shape
        data = torch.reshape(data, (1, 1, n2, n3))
        # target4 = torch.reshape(target4, (1, 1, n2, n3))
        target5 = torch.reshape(target5, (1, 1, n2, n3))

        model.set_input(data, target5)  # unpack data from data loader
        start_time = time.time()
        model.test()  # run inference
        # end_time = time.time()
        # duration = start_time - end_time

        test_image(model.real_A[0].cpu(), model.fake[0].cpu(), model.real_B[0].cpu(), xlims, zlims,i ,opt.phase,opt.name)
        img_eva.evaluate(model.fake[0].cpu().detach().numpy(), model.real_B[0].cpu().detach().numpy(),opt,plane_wave_data, test_type, i)
        # img_eva.evaluate(target4[0].cpu().detach().numpy(), target5[0].cpu().detach().numpy(), opt,
        #                  plane_wave_data, test_type, i)
        # test_image(data[0].cpu(), target1[0].cpu(), target5[0].cpu(), xlims, zlims, i, opt.phase)

        i = i + 1
    img_eva.print_results(opt) # Print the results