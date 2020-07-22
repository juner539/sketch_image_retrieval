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
        parser.add_argument('--dataset', required=True, default='Sketchy',
                            help='Name of the dataset, choices= [sketchy, TU_Berlin, Quick_Draw ]')
        # image dir
        parser.add_argument('--photo_dir', required=True, default='C:/Users/xiangjun/datafolder/sketch_image/rendered_256x256/256x256/photo',
                            help='The directory for photo')
        # sketch dir
        parser.add_argument('--sketch_dir', required=True, default='C:/Users/xiangjun/datafolder/sketch_image/rendered_256x256/256x256/photo',
                            help='The directory for sketch')



        self.parser = parser

    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()
