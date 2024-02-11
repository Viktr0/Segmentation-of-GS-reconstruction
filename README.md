# Semantic segmentation

There are many solutions to the 3D reconstruction problem, but a major breakthrough was the NeRF method, which does not deal with the explicit description of the surface of the shapes, but treats the space as a colorful, dense, luminous volume, called radiance field. This approach is extremely flexible, unlike previous methods, it can learn viewpoint-dependent lighting without any knowledge of the light source.

<p align="center">
  <img width=70% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/e4a93227-dfd4-4537-a602-91393c0e99eb" alt="animated" />
</p>

The radiance field approaches space as particles, as in reality. The radiance field can be represented using a neural network, this is the NeRF method. However, according to another approach, these particles can also have an extent. These extended particles are Gaussian splats which are geometrically equivalent to the ellipsoid.

Rendered images of the reconstruction             |  Ellipsoids of the reconstruction
:-------------------------:|:-------------------------:
![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/9d4e75e4-2702-4ed5-bc14-1161d3cd1abd)  |  ![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/8d4dc012-cd28-47df-9191-8b02bf619785)

The ellipsoids are stretched by Gaussian distributions, by 3 mutually perpendicular axes. Similarly, they have the necessary attributes, such as viewpoint-dependent color and transparency.
The method that based on this concept called 3D Gaussian Splatting.

## Output of Gaussian Splatting

Gaussian Splatting is a method for representing 3D scenes and allows to render radiance fields in real-time.
Gaussian Splatting is capable of generating the 3D reconstruction of a scene by only using images and the corresponding camera poses, based on machine learning algorithms.
The method optimises the attributes of the ellipsoids by comparing the rendered and a real images from given camera poses.

The output of the algorithm is a Polygon file and it contains the attributes of the ellipsoids that build up the reconstruction:
* center
* rotation
* color (diffuse and view dependent components)
* scale
* opacity

<p align="center">
  <img width=40% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/bdd29a76-2142-4e90-ae9a-3f7d69bc2c60" />
</p>

## The program

Segmentation is a commonly used procedure. 
One way to solve the segmentation of the 3D scene is to project the segmented images to the points (ellipsoids) of the reconstructed scene.
For this method you need the semantic segmented version of the images of the training dataset, which the reconstruction is based on.
It is important to name the semantic segmented version of the images as same as their original RGB pairs to know where to project from in the world coordinate system.

RGB             |  Semantic segmentation
:-------------------------:|:-------------------------:
![58](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/d83c6ba7-5d41-45e8-8257-11ba68faef2e)  |  ![58_semseg](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/bbc1dbad-672e-4d1d-8fee-cd13c1565f89)


### Ray-casting
For the projection I implemented an OO code in Python based on ray-casting. Here you can see the partial class diagram of the implementation.

![classdiagram](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/e9bda5f3-b07d-4498-9970-138bc0a9bb66)

The code takes the camera poses and the corresponding images, than intersects the ellipsoids by the ray that goes to the direction of a given pixel coordinate and labels the closest ellipsoid with the color of the pixel.

The implementation contains functions that allows you to
* label the ellipsoids
* smooth the segmentation
* reduce the noise of the segmentation
* filter by labels
* and save the results as the original format

## Results

For the visualization of the partial results I used the Open3D library.

![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/95b0bc25-9cdd-4c3a-a1cd-500aa2622e0f)

This project is capable of saving a properly formatted Polygon file which can be read in by the SIBR viewer. This way you can eliminate the unnecessary segments of the scene after labeling it. You can see the results below.

<p align="center">
  <img width=80% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/40c1baf7-f27c-4588-87c2-131e8abc7731" />
</p>

<p align="center">
  <img width=80% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/7f6f15e8-23e3-4d59-99b7-ba7ca9b6ad3d" />
</p>

<p align="center">
  <img width=80% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/63b6f45d-e5d5-428f-9c58-9d74ecd38aa2" />
</p>
