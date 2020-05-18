"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time
import numpy as np
import cv2
from PIL import Image
import scipy.io as sio
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_numpy, remove_dataparallel_wrapper, is_rgb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def test_ffdnet(**args):
	r"""Denoises an input image with FFDNet
	"""
	# Init logger
	logger = init_logger_ipol()

	# Check if input exists and if it is RGB
	try:
		rgb_den = is_rgb(args['input'])
	except:
		raise Exception('Could not open the input image')

  # Measure runtime
	start_t = time.time()

	# Open image as a CxHxW torch.Tensor
	if rgb_den:
		in_ch = 3
		model_fn = 'net_rgb.pth'
		imorig = Image.open(args['input'])
		imorig = np.array(imorig, dtype=np.float32).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		in_ch = 1
		model_fn = 'models/net_gray.pth'
		# imorig = cv2.imread(args['input'], cv2.IMREAD_GRAYSCALE)
		imorig = np.expand_dims(imorig, 0)
	imorig = np.expand_dims(imorig, 0)

	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = imorig.shape
	if sh_im[2]%2 == 1:
		expanded_h = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

	if sh_im[3]%2 == 1:
		expanded_w = True
		imorig = np.concatenate((imorig, \
				imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)

	imorig = normalize(imorig)
	imorig = torch.Tensor(imorig)
	# Absolute path to model file
	model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
				model_fn)

	# Create model
	print('Loading model ...\n')
	net = FFDNet(num_input_channels=in_ch)

	# Load saved weights
	if args['cuda']:
		state_dict = torch.load(model_fn)
		#device_ids = [0,1,2,3]
		#model = nn.DataParallel(net, device_ids=device_ids).cuda()
		#state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	else:
		state_dict = torch.load(model_fn, map_location='cpu')
		# CPU mode: remove the DataParallel wrapper
		state_dict = remove_dataparallel_wrapper(state_dict)
		model = net
	model.load_state_dict(state_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model.eval()
	

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		dtype = torch.cuda.FloatTensor
	else:
		dtype = torch.FloatTensor

  # Test mode
	with torch.no_grad(): # PyTorch v0.4.0
		imorig = Variable(imorig.type(dtype))
		nsigma = Variable(
				torch.FloatTensor([args['noise_sigma']]).type(dtype))
	

	# # Measure runtime
	# start_t = time.time()

	# Estimate noise and subtract it to the input image
	im_noise_estim = model(imorig, nsigma)
	stop_t = time.time()

	# log time
	if rgb_den:
		print("### RGB denoising ###")
	else:
		print("### Grayscale denoising ###")
	print("\tRuntime {0:0.4f}s".format(stop_t-start_t))

	# Save noises
	noise = variable_to_numpy(imorig.to(3) - im_noise_estim).transpose(1, 2, 0)
	filename = args['input'].split('/')[-1].split('.')[0]
	if args['save_path']:
		sio.savemat('./output_noise/'+args['save_path']+'/'+filename+'.mat',{'Noisex':noise})
	else:
		sio.savemat('./output_noise/'+filename+'.mat',{'Noisex':noise})
	

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="FFDNet_Test")
	parser.add_argument("--input", type=str, default="", \
						help='path to input image')
	parser.add_argument("--save_path", type=str, default="", \
						help='suffix to add to output name')
	parser.add_argument("--noise_sigma", type=float, default=5, \
						help='noise level used on test set')
	parser.add_argument("--no_gpu", action='store_true', \
						help="run model on CPU")
	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_ffdnet(**vars(argspar))
