import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk  
import os
import copy
import sys

class LocalisedElasticDeformation():

	"""Bspline code adapted from [https://gist.github.com/fepegar/b723d15de620cd2a3a4dbd71e491b59d]"""

	def __init__(self, image_list, mask_list, cfg):
		self.image_list = sorted(image_list)
		self.mask_list = sorted(mask_list)
		self.num_files = len(image_list)
		self.data_dir = cfg.data_dir
		self.results_dir = cfg.results_dir
		self.enlarge = cfg.enlarge
		self.noise_amount = cfg.noise_amount
		self.radius = cfg.radius
		self.max_displacement = cfg.max_displacement
		self.ctrl_pts = np.array(cfg.ctrl_pts, np.uint32)
		self.mesh_size = self.ctrl_pts - cfg.spline_order
		self.grid_shape = *self.ctrl_pts, 2

	def run(self):

		print('Starting Localised Elastic Deformation on {} file(s)..'.format(self.num_files))
		for i in range(self.num_files):
			id_ = self.image_list[i].split('/')[-1].split('_')[0]
			img = np.load(self.image_list[i])
			mask = np.load(self.mask_list[i])
			self.deform_one_example(img, mask, id_)
		print('Augmentation Complete')


	def mask_stats(self, mask):
		"""stats about the mask"""
		x, y = np.where(mask > 0)
		mask_centre = np.mean(y), np.mean(x)
		width = len(np.unique(y))
		height = len(np.unique(x))
		return mask_centre, width, height


	def deform_one_example(self, img, mask, id_):

		mask_centre, width, height = self.mask_stats(mask)

		img_sitk = sitk.GetImageFromArray(img) 
		mask_sitk = sitk.GetImageFromArray(mask) 

		# initialise bspline 
		transform = sitk.BSplineTransformInitializer(img_sitk, self.mesh_size.tolist())
		x_coeff, y_coeff = transform.GetCoefficientImages()
		grid_origin = x_coeff.GetOrigin()
		grid_spacing = x_coeff.GetSpacing()

		# create grid
		grid_coords, xx, yy = self.create_grid(self.ctrl_pts, grid_origin, grid_spacing)

		# find grid centre
		grid_centre, i, j = self.calc_grid_centre(grid_coords, mask_centre)

		# directional vectors from grid_centre to grid points
		dir_vec = grid_coords - grid_centre

		# add a bit of noise to the vectors
		dir_vec *= (np.random.rand(*self.grid_shape) * self.noise_amount) + sys.float_info.epsilon

		# zero all points unless they are within `radius` number of points away from grid centre
		uv = np.zeros(dir_vec.shape)
		uv[i-self.radius:i+self.radius, j-self.radius:j+self.radius] = dir_vec[i-self.radius:i+self.radius, j-self.radius:j+self.radius]

		# scale vectors
		max_uv = np.max(np.abs(uv))
		if self.enlarge == True: 
			uv = uv * - (self.max_displacement / max_uv) # negative value enlarges the area, 
		else: 
			uv = uv * (self.max_displacement / max_uv) # positive value will decrease it. 

		# do deformation
		img_bspline = self.bpsline(img_sitk, uv)
		mask_bspline = self.bpsline(mask_sitk, uv)

		# visualise and save output
		self.visualise([img, mask], [img_bspline,mask_bspline], uv, xx, yy, id_)
		np.save(self.results_dir + '/' + id_ + '_aug_image.npy', img_bspline)
		np.save(self.results_dir + '/' + id_ + '_aug_mask.npy', mask_bspline)

	def visualise(self, inputs, outputs, uv, xx, yy, id_):

		fig, axes = plt.subplots(2, 2, figsize=(10,10), dpi=150)
		all_ = inputs+outputs
		for i, ax in enumerate(axes.reshape(-1)):
			ax.grid()
			ax.imshow(all_[i],interpolation='hamming')
			u, v = uv[..., 0].T, uv[..., 1].T
			ax.scatter(xx, yy, s=1);
			if i >=2:
				ax.quiver(xx, yy, -u, -v, color='red',units='xy', angles='xy', scale_units='xy', scale=1)
		plt.savefig(os.path.join(self.results_dir, id_ + '_plot.png'))

	def bpsline(self, x, uv):
		# do bspline
		transform = sitk.BSplineTransformInitializer(x, self.mesh_size.tolist())
		transform.SetParameters(uv.flatten(order='F').tolist())
		resampler = sitk.ResampleImageFilter()
		resampler.SetReferenceImage(x)
		resampler.SetTransform(transform)
		resampler.SetInterpolator(sitk.sitkLinear)
		resampler.SetDefaultPixelValue(0.5)
		resampler.SetOutputPixelType(sitk.sitkFloat32)
		resampled = resampler.Execute(x)
		result = copy.deepcopy(sitk.GetArrayViewFromImage(resampled))
		return result

	def calc_grid_centre(self, grid_coords, mask_centre):
		# find the grid point which is closest to our mask centre. 
		distance_to_mask_centre = np.sum(np.abs(grid_coords - mask_centre), axis=-1) 
		i, j = np.where(distance_to_mask_centre == distance_to_mask_centre.min())
		i, j = i[0], j[0]
		grid_centre = grid_coords[i, j,:]
		return grid_centre, i, j

	def create_grid(self, ctrl_pts, grid_origin, grid_spacing):
		# create grid coordinates
		x = np.linspace(grid_origin[0], grid_origin[0] + (ctrl_pts[0] - 1) * grid_spacing[0], ctrl_pts[0])
		y = np.linspace(grid_origin[1], grid_origin[1] + (ctrl_pts[1] - 1) * grid_spacing[1], ctrl_pts[1])
		xx, yy = np.meshgrid(x, y)
		grid_coords = np.stack((yy, xx),-1)
		return grid_coords, xx, yy
