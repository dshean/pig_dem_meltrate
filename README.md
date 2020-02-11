# pig_dem_meltrate

David Shean  
November 2016

Processing scripts used to prepare DEM data for analysis of Pine Island Glacier melt rates as described in the following papers:
* Shean, D. E., I. R. Joughin, P. Dutrieux, B. E. Smith, and E. Berthier (2019), Ice shelf basal melt rates from a high-resolution digital elevation model (DEM) record for Pine Island Glacier, Antarctica, The Cryosphere, 13(10), 2633–2656, doi:10.5194/tc-13-2633-2019.
* Shean, D. E., K. Christianson, K. M. Larson, S. R. M. Ligtenberg, I. R. Joughin, B. E. Smith, C. M. Stevens, M. Bushuk, and D. M. Holland (2017), GPS-derived estimates of surface mass balance and ocean-induced basal melt for Pine Island Glacier ice shelf, Antarctica, The Cryosphere, 11(6), 2655–2674, doi:10.5194/tc-11-2655-2017.

See `stack_proc.sh` for an overview of the full DEM stack correction and melt rate calculation workflow

Input was stack of 256 m gridded DEMs over the PIG catchment, generated using `make_pig_stack_20160307.sh`

Basic workflow: 
1. `ndinterp.py` performs optimization to solve for DEM tilt and bias correction
2. `apply_lsq_tiltcorr.py` applies the resulting corrections to the DEMs in the stack
3. `ndinterp_vel.py` prepares interpolated velocity stacks for melt rate calculation
4. `stack_melt_lsq_path.py` computes melt rates
5. `ndinterp_dem.py` prepares interpoated DEM stacks for flux-gate analysis

I apologize for the states of these scripts - most were written and re-written during the frantic 3-6 months before my PhD defense, when I had many bad habits.  I had every intention of cleaning these up and releasing, but other projects quickly occupied my time.  Maybe someday...
