The code in this repository utilizes data from the PHANGS-MUSE Nebular catalog and data obtained from the SITELLE spectrograph at the CFHT to directly calculate metallicities in HII regions of nearby galaxies. I present code from this project in the Framework_code file and the Sandboxes file. The code used to produce final results are shown in the Framework_code file and the code used to make plots and refine my methods are presented in Sandboxes.


This project has three main components: the extraction of spectra from the SITELLE data cube, the spectral fitting of the [OII]3727,7319,7330, [NII]5755 and [SIII]6312 features, and the derivation of physical quantities.

spectra_pipeline.ipynb The file extracts spectra from the SITELLE data cubes. This is done using the ORBS Python environment and package, which is the analysis software used for SITELLE observations. The notebook also extracts a background spectrum that is scaled to the number of pixels in each HII regions. The extracted spectra, the wavenumber axis, and the background spectra are all saved.

MUSE+SITELLE_flux_Pipeline.ipynb This file contains the line fitting code for MUSE-PHANGS galaxies with SITELLE data. Faint emission lines, also known as auroral lines, as well as the prominent [OII]3727 flux, known as a nebularl line, are fitted to obtain flux values. The four auroral lines are [OII]7319,7330, [NII]5755, and [SIII]6312. Orgininally the background continuum was not subtracted in the Nebular catalog, so in this notebook I fit the auroral lines with a gaussian with a constant offset to account for the background continuum. The [OII]3737 is fitted using a sinc function provided by the ORBS analysis software. The error in these measurements are calculated using monte carlo techniques.

INIT_PQ_Pipeline.ipynb This file uses PyNeb to compute physical quantities using my refitted auroral lines, the nebular lines from the PHANGS Nebular catalog and the [OII]3727 fluxes from SITELLE. These directly derived physical quantities include [NII], [OII], [SIII] and [OIII] temperatures, [SII] densities, and both elemental and ionic oyxgen abundances.


These files are all test code and may be useful to trying to understand the packages used in this analysis.

PhysQuans_sandbox.ipynb This file is used to test different diagnostics for line ratios in PyNeb and to test how different methods of the extraction of the [OII]3727 feature affect the resultant physical quantities.

development_Sandbox.ipynb This file is used to understand how to use the ORBS analysis pacakge and to learn how to interact with the SITELLE data cube. The code that is used to extract spectra from the SITELLE data cube stems from this notebook.

SkyBackground_sandbox.ipynb This file is used to test how to extract a sky background spectrum on a per pixel basis using the ORBS analysis package and how to subtract the sky background from an HII region spectrum, that can then be used to extract an [OII]3727 flux.

fitting_sandbox.ipynb This file tests using the ORBS analysis package to fit the [OII]3727 feature. Initially we had hoped to fit both the [OII]3726,3729 features, though we found this to be quite difficult and did not produce reliable flux or temperature values. We see code to fit both of the [OII]3726,3729 features as well as the blended [OII]3727 feature. We also see my results compared to another member of the PHANGS collaboration.

SIGNALS_comparison.ipynb This file compares my results to the results from the SIGNALS survey: NGC628 with SITELLE: I. Imaging spectroscopy of 4285 H II region candidates.