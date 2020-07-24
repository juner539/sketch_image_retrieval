# -*- coding: utf-8 -*-
"""
    Parse input arguments
"""

import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='Zero-Shot Sketch-based Image Retrieval.')
        # Optional argument
        parser.add_argument('--dataset', default='Sketchy',
                            help='Name of the dataset, choices= [sketchy, TU_Berlin, Quick_Draw ]')
        # image dir
        parser.add_argument('--photo_dir', default='/home/xiangjun/dataset/sketch_database/sketch_image/rendered_256x256/256x256/photo',
                            help='The directory for photo')
        # sketch dir
        parser.add_argument('--sketch_dir', default='/home/xiangjun/dataset/sketch_database/sketch_image/rendered_256x256/256x256/sketch',
                            help='The directory for sketch')

        parser.add_argument('--batch_size',
                            default=16, help='batch_size')
        parser.add_argument('--lr',
                            default=1e-3, help='learning rate')
        parser.add_argument('--weight_decay',
                            default=4e-5, help='weight_decay')
        parser.add_argument('--epoch',
                            default=100, help='epoch')



        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
