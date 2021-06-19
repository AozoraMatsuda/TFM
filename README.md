# Traction Force Microscopy
This is a python libarary for traction force microscopy. **pykalman and pyimagej is needed.**
## Vector field class
- Vectors is the parent class of DPF and TFF
- DPF is a class for DisPlacement Field.
- TFF is a class for Traction Force Field.
## Vectors
- generate_fields : generate a list of vector fileds by artificial means
- draw : plot vector field data
## DPF
- load_PIV : load piv data (.txt) and return DPF object
- fftc : estimate traction force field by FFTC
## TFF
- load_TFF : load traction force field data (.txt) and return TFF object
- inv_fftc : calculate displacement filed by FFTC
- kalman_FFTC : estimate vector field series by kalman smoother and FFTC
