"""
Functions implementing custom NN layers

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
from torch.autograd import Function, Variable

def concatenate_input_noise_map(input, noise_sigma):
	r"""Implements the first layer of FFDNet. This function returns a
	torch.autograd.Variable composed of the concatenation of the downsampled
	input image and the noise map. Each image of the batch of size CxHxW gets
	converted to an array of size 4*CxH/2xW/2. Each of the pixels of the
	non-overlapped 2x2 patches of the input image are placed in the new array
	along the first dimension.

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
