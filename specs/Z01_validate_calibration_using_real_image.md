# Z01_validate_calibration_using_real_images

## Purpose
Guide the user through validating calibration using a real calibration image and a real dark image, leveraging existing repository subroutines for image loading, metadata extraction, ROI selection, center finding, annular reduction, peak finding, and Tolansky analysis.

## Steps

### (a) Use Existing Subroutines
- All image processing, center finding, reduction, and analysis steps must use subroutines already implemented in the repository.

### (b) User Prompt: Select Images
- Prompt the user to select:
  - A real calibration image file
  - A real dark image file
- Load both images using existing image loading routines.

### (c) Extract Metadata
- Extract metadata from both images.
- Copy out the first row of metadata from each image for display.

### (d) Display Raw Images and Metadata
- Create a figure showing the raw calibration image and raw dark image side by side.
- Display a table below the figure with the extracted metadata for both images.

### (e) User-Guided ROI Selection
- Prompt the user to visually select the fringe pattern center using mouse/cursor.
- Use this center to define a default 216 x 216 ROI (ask user to confirm or adjust dimensions).
- Extract this ROI from both the calibration and dark images for further analysis.

### (f) Display ROI and Histograms
- Create a figure showing the calibration and dark ROIs side by side.
- Display histograms of pixel values for each ROI below or alongside the images.

### (g) Dark Subtraction
- Subtract the dark ROI array from the calibration ROI array to produce a dark-subtracted image.

### (h) Center Finding
- Process the dark-subtracted image using the center_finder routines:
  - Coarse grid variance analysis
  - Fine grid Nelder-Mead variance analysis
- Both techniques must be used as implemented in the repo.

### (i) Center Finding Results Visualization
- Create a figure showing results for both center finding techniques:
  - Plots of calculated variance as a function of center pixel position (chi-square)
  - Display the final most accurate center position, uncertainties, and chi-square results

### (j) Annular Reduction and Peak Finding
- Feed the determined center position into the annular_reduction/peak finder routines.
- Identify all 20 neon calibration lamp peaks (10 for 640 nm, 10 for 638 nm).
- Create a figure of the radial profile average with SEM error bars.
- Overlay peak finding results (positions, uncertainties, arrows pointing to peaks).
- Below the figure, display a table of all 20 peaks with full fit results and 2-sigma uncertainties.

### (k) Tolansky Analysis
- Feed the peak fit results into the Tolansky analysis routines.
- Produce one figure combining the 1-line analysis for both 640 nm and 638 nm lines, highlighting the fringe fractional result (epsilon) at line center for both.
- Produce a separate figure for the joint 2-line analysis, showing focal length and etalon gap, with uncertainties.

---

**Note:** This spec defines the workflow and required outputs. Do not implement yet—review and confirm the spec before proceeding to code.
