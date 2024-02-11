# Semantic segmentation

3D rekonstrukciós problémára számos megoldás létezik, viszont egy komoly áttörést jelentett a NeRF módszer, mely nem foglalkozik az alakzatok felületének explicit leírásával, hanem a teret egy színes, sűrű, világító, közegként kezeli, ez a radiancia mező. Ez a megközelítés rendkívül rugalmas, a korábbi módszerekkel ellentétben képes megtanulni a nézőpont függő megvilágítást, anélkül, hogy a fényforrásról ismereteink lennének.

<p align="center">
  <img width=70% src="https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/e4a93227-dfd4-4537-a602-91393c0e99eb" alt="animated" />
</p>

NeRF

A radiancia mező a teret, a valósághoz hasonlóan részecske szinten közelíti meg, színes sűrű közegként. A radiancia mező reprezentálható egy neurális hálózat segítségével, ez a korábbi fejezetben részletezett NeRF. Viszont egy másik megközelítés szerint ezeknek a részecskéknek lehet kiterjedése is. Ezek a kiterjedéssel rendelkező részecskék a Gauss-halmazok, melyek geometriája megfelel az ellipszoidéhoz.

Az ellipszoidokat Gauss eloszlások feszítik ki, értelemszerűen 3 egymásra merőleges tengely mentén. Hasonlóan rendelkeznek a szükséges attribútumokkal, ilyen a nézőpont függő szín és áttetszőség. 
perpendicular
This conception is called Gaussian Splatting
Ezt a koncepciót feldolgozó eljárás a Gaussian Splatting.

Rendered iages of the reconstruction             |  Ellipsoids of the reconstruction
:-------------------------:|:-------------------------:
![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/9d4e75e4-2702-4ed5-bc14-1161d3cd1abd)  |  ![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/8d4dc012-cd28-47df-9191-8b02bf619785)


## Output of Gaussian Splatting

GS can reconstruct using only images of the scene and the corresponding camera poses.

output file contains the ellipsoids of the scene

attribútumok

KEP ellipsoids
![image](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/bdd29a76-2142-4e90-ae9a-3f7d69bc2c60)


## The program

Segmentation is a commonly used procedure. 
One way to solve the segmentation of the 3D scene is to project the segmented images to the points (ellipsoids) of the reconstructed scene.
For this method you need the semantic segmented version of the images of the training dataset, which the reconstruction is based on.
It is important to name the semantic segmented version of the images as same as their original RGB pairs to know where to project from in the world coordinate system.

RGB             |  Semantic segmentation
:-------------------------:|:-------------------------:
![58](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/d83c6ba7-5d41-45e8-8257-11ba68faef2e)  |  ![58_semseg](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/bbc1dbad-672e-4d1d-8fee-cd13c1565f89)


### Ray-casting
For the projection I implemented an OO code in Python based on ray-casting

![classdiagram](https://github.com/Viktr0/Segmentation-of-GS-reconstruction/assets/47856193/e9bda5f3-b07d-4498-9970-138bc0a9bb66)




## Results
