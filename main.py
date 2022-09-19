import os
import argparse
from glob import glob
from elastic_deformation import LocalisedElasticDeformation

def main(config):

	# file list
	images = glob(config.data_dir + '/*_image.npy')
	masks = glob(config.data_dir + '/*_mask.npy')
	assert len(images) == len(masks)

	# do deformation
	augmentor = LocalisedElasticDeformation(images, masks, config)
	augmentor.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories.
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--results_dir', type=str, default='results/')
    parser.add_argument('--enlarge', type=bool, default='True', help='Mask increase or decrease in size?')
    parser.add_argument('--max_displacement', type=int, default=15, help='max displacement of ctrl point')
    parser.add_argument('--radius', type=int, default=2, help='radius of ctrl points that will be displaced')
    parser.add_argument('--noise_amount', type=int, default=1, help='how much noise to add to displacement vectors')
    parser.add_argument('--ctrl_pts', type=tuple, default=(15, 15), help='num of ctrl points')
    parser.add_argument('--spline_order', type=int, default=3)

    config = parser.parse_args()
    main(config)