# SEMdenoising
Our paper has been published and can be found here: https://iopscience.iop.org/article/10.1088/1361-6501/ad7e41/meta 

If you have any questions, please contact: seoleuns@kriss.re.kr

## License

All files in this repository, including the TIFF images under `Data/`, are covered by the MIT license.

## Acquisition Details

- **Instrument**: FEI Quanta-FEG 650 (field emission SEM)
- **Acceleration voltage**: 10 kV
- **Working distance**: 10 mm
- **Detector**: ETD, SE mode
- **Beam current**: Not measured in absolute terms; controlled via the instrument's "Spot" parameter (Spot = 3)
- **Gain**: No separate gain control other than the standardized "contrast" knob (digital gain = 1)
- **Field of view**: 2.984 µm (H) × 2.576 µm (V); pixel size 5.83 nm/pixel
- **Averaging scheme**: Clean images were acquired by scanning each line 50 times and averaging before proceeding to the next line. Noisy images were scanned once per line (no averaging). Nothing else was changed between paired acquisitions.
- **Acquisition order**: In each pair, the noisy image was acquired first.
- **Registration**: No registration/alignment was applied. All images are raw originals as acquired.

## Dataset Structure

- **Indices 001–050**: Independent fields of view. Imaging locations were selected freely, sufficiently far apart from each other.
- **Pairing**: Images with the same numeric suffix were taken at the same physical field of view (applies to both low-dwell/high-dwell pairs and line-off/line-50× pairs).

## Image Format

- **TIFF size**: 512 × 471 pixels
  - Rows 0–441: the 512 × 442 SEM image field
  - Rows 442–470: instrument information footer (exclude from analysis)
- **Bit depth**: 8-bit (pixel values 0–255)
- **Colored pixels**: Green/yellow pixels are instrument overlays (scale bar etc.) and may be excluded from quantitative analysis.

## Citation

If you use this dataset, please cite:

S Shin, IH Lee, BC Park, JH Song: Applications of deep learning-based denoising methodologies for scanning electron microscope images, Measurement Science and Technology, 2025
[DOI 10.1088/1361-6501/ad7e41]

and link to this repository.
