import torch
from torch.autograd import Function, Variable

def concatenate_input_noise_map(input, noise_sigma):
	r"""Implements the first layer of FFDNet. This function returns a
	torch.autograd.Variable composed of the concatenation of the downsampled
	input image and the noise map.

	Args:
		input: batch containing CxHxW images
		noise_sigma: the value of the pixels of the CxH/2xW/2 noise map
	"""
	# noise_sigma is a list of length batch_size
	N, C, H, W = input.size()
	# Fill the downsampled image with zeros

	# Build the CxH/2xW/2 noise map
	noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, 1, H, W)

	# concatenate de-interleaved mosaic with noise map
	return torch.cat((input, noise_map), 1)
