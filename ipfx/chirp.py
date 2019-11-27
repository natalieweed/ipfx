from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import logging
import pandas as pd
import scipy.fftpack as fftpack

from ipfx.error import FeatureError
from . import offpipeline_utils as op
from . import feature_vectors as fv
from . import time_series_utils as tsu

CHIRP_CODES = [
            "C2CHIRP180503", # current version, single length
            "C2CHIRP171129", # early version, three lengths
            "C2CHIRP171103", # only one example
        ]

def extract_chirp_features(specimen_id, data_source='lims', sweep_qc_option='none', method_params={}):
    try:
        dataset = op.dataset_for_specimen_id(specimen_id, data_source=data_source, ontology=ontology_with_chirps())
        sweepset = op.sweepset_by_type_qc(dataset, specimen_id, stimuli_names=["Chirp"])
    except:
        logging.warning("Error loading data for specimen {:d}.".format(specimen_id), exc_info=True)
        return {}

    results = []
    for sweep in sweepset.sweeps:
        try:
            results.append(chirp_sweep_features(sweep, method_params=method_params))
        except FeatureError as exc:
            logging.debug(exc)
        except Exception:
            msg = "Error processing chirp sweep {} for specimen {:d}.".format(sweep.sweep_number, specimen_id)
            logging.warning(msg, exc_info=True)

    if len(results)==0:
        logging.debug("No chirp sweep results for specimen {:d}.".format(specimen_id))
        return {}

    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    return mean_results

def chirp_sweep_amp_phase(sweep, method_params={}):
    v, i, freq = transform_sweep(sweep, **method_params)
    Z = v / i
    amp = np.abs(Z)
    phase = np.angle(Z)

    from scipy.signal import savgol_filter
    # pick odd number, approx number of points for 2 Hz interval
    n_filt = int(np.rint(1/(freq[1]-freq[0])))*2 + 1
    filt = lambda x: savgol_filter(x, n_filt, 5)
    amp, phase = map(filt, [amp, phase])
    return amp, phase, freq

def chirp_sweep_features(sweep, method_params={}):
    amp, phase, freq = chirp_sweep_amp_phase(sweep, method_params=method_params)
    i_max = np.argmax(amp)
    z_max = amp[i_max]
    i_cutoff = np.argmin(abs(amp - z_max/np.sqrt(2)))
    features = {
        "peak_ratio": amp[i_max]/amp[0],
        "peak_freq": freq[i_max],
        "3db_freq": freq[i_cutoff],
        "z_low": amp[0],
        "z_high": amp[-1],
        "z_peak": z_max,
        "phase_peak": phase[i_max],
        "phase_low": phase[0],
        "phase_high": phase[-1]
    }
    return features

def extract_chirp_feature_vector(data_set, chirp_sweep_numbers):
    chirp_sweeps = data_set.sweep_set(chirp_sweep_numbers)

    results = []
    for sweep in chirp_sweeps.sweeps:
        try:
            results.append(chirp_sweep_features(sweep, method_params={}))
        except FeatureError as exc:
            logging.debug(exc)
        except Exception:
            msg = "Error processing chirp sweep {} for specimen {:d}.".format(sweep.sweep_number, specimen_id)
            logging.warning(msg, exc_info=True)

    if len(results)==0:
        logging.debug("No chirp sweep results for specimen {:d}.".format(specimen_id))
        return {}

    mean_results = {}
    mean_results = {key: np.mean([res[key] for res in results]) for key in results[0]}
    
    result = {}
    result = np.hstack(results)

    return result, mean_results



def transform_sweep(sweep, n_sample=10000, min_freq=1., max_freq=35.):
    sweep.select_epoch("stim")
    if np.all(sweep.v[-10:] == 0):
        raise FeatureError("Chirp stim epoch truncated.")
    v = sweep.v
    i = sweep.i
    t = sweep.t
    N = len(v)

    # down_rate=2000
    # width = int(sweep.sampling_rate / down_rate)

    width = int(N / n_sample)
    pad = int(width*np.ceil(N/width) - N)
    v = fv._subsample_average(np.pad(v, (pad,0), 'constant', constant_values=np.nan), width)
    i = fv._subsample_average(np.pad(i, (pad,0), 'constant', constant_values=np.nan), width)
    t = t[::width]

    N = len(v)
    dt = t[1] - t[0]
    xf = np.linspace(0.0, 1.0/(2.0*dt), N//2)

    v_fft = fftpack.fft(v)
    i_fft = fftpack.fft(i)

    low_ind = tsu.find_time_index(xf, min_freq)
    high_ind = tsu.find_time_index(xf, max_freq)

    return v_fft[low_ind:high_ind], i_fft[low_ind:high_ind], xf[low_ind:high_ind]


