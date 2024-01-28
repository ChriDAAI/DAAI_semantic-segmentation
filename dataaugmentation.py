#!/usr/bin/python
# -*- encoding: utf-8 -*-


from PIL import Image
import random
import numpy as np
from torchvision import transforms


class DataAugmentation(object):

    def __call__(self, image, label):
        method= np.random.choice([RandCrop(), HorizontalFlip(), Jitter()])
        return method(image, label)


class RandCrop(DataAugmentation):
	"""Crop the given Image and label at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
	"""
	def __init__(self, seed=42): 				# void class for padding
		super(RandCrop, self).__init__()
		self.size = None
		random.seed(seed)


	@staticmethod
	def get_params(img, output_size, h, w):
		tw, th = output_size
		if w == tw and h == th:
			return 0, 0, h, w
		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return (j, i, j+tw, i+th)


	def __call__(self, img, label):
		#with PIL as input
		w, h =label.size
		self.size = w//2, h//2

		crop_box= self.get_params(label, self.size, h, w)

		cropped_image = img.crop(crop_box).resize((w, h), Image.NEAREST)	
		cropped_label = label.crop(crop_box).resize((w, h), Image.NEAREST)	

		return cropped_image, cropped_label
    
	
	def __str__(self):
		return "Random crop"


class HorizontalFlip(DataAugmentation):
	"""Horizontal Flip of the given Image and label.

	Args:
				-
	"""
	def __init__(self):
		super(HorizontalFlip, self).__init__()

	def __call__(self, img, label):
		flipped_image = img.transpose(Image.FLIP_LEFT_RIGHT)
		flipped_label = label.transpose(Image.FLIP_LEFT_RIGHT)
		return flipped_image, flipped_label
	
	def __str__(self):
		return "Horizontal flip"
	
	
	
class Jitter(DataAugmentation):
	def __init__(self):
		super(Jitter).__init__()

	def __call__(self, img, label):
		jitter_transform = transforms.Compose([
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
		])
		# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
		jittered_image = jitter_transform(img)
		return jittered_image, label
	
	def __str__(self):
		return "Jitter"
