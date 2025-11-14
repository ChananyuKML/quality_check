
# Holes Inspector

This repository was made in order to compare the each components in object in an image to the reference one. To archieve this goal, the process is divide to 2 main parts which are reprojection and similarity comparision using cosine similary.


## Installation

Create environment for containing necessary dependencies

```bash
  python -m venv .venv
  source .venv/bin/activate
```
### Install dependencies

```bash
  pip install torch torchvision opencv-python Pillow
```
## Usage
![screenshot](img/pc.jpg)
### Reprojection
Before a new product can be compared to the "golden" reference, it must be perfectly aligned. Products on a conveyor belt or in a testing jig are never in the exact same position, rotation, or scale.

This process needs corresponding points between the images. These points are used to calculate a homography matrix, which mathematically describes the perspective distortion. This matrix is then fed into cv2.warpPerspective to transform the "taken image" so it perfectly overlays the reference image

```bash
python src/reprojection.py --img <your-image-name>
```

After execute the code, the input image will be shown. Corners of the side that will be inspected are needed to be identified by left-clicking on each corner of an object in image. The order of clicking need to be the same as sequence below

```bash
top-left > top-right > bottom-right > bottom-left
```
Then the output of reprojected image will be saved as front_view_ourput.jpg. The side in file name can be specifed by add an option to the executing command line.

```bash
python src/reprojection.py --img <your-image-name> --side <your-reprojected-side>
```

### Similarity Comparision
Once the images are aligned, the tool performs a detailed, feature-based comparison to find discrepancies. Instead of simple pixel-matching, it uses a deep learning model to "understand" the content of each hole. Comparing two images by extracting features in an image using vision transformer(ViT) then compare the extracted vectors with cosine similarity.

![screenshot](img/front_view_output.jpg)
![screenshot](img/front_ref.jpg)

```bash
python src/similarity_vit.py <path-to-image> <path-to-reference-image>
```
![screenshot](img/sim_output.png)

## Bounding Box comparison
In the previous process, whole image have been reprojected and compare to the reference. To compare the specific area in the image, user need to create bounding bom data on the reference image using label studio(yolo format), then apply the box similaryty comparison using this command.

```bash
python src/read_box.py <image-to-test> --ref <path-to-ref-image> --boxes <path-to-label-file>
```

### Reference-box VS Reprojected-box
![screenshot](img/reference_box.png)
![screenshot](img/reprojected_box.png)
### Similarity Scores
![screenshot](img/Pasted image.png)
