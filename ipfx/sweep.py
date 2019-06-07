import ipfx.epochs as ep


class Sweep(object):
    def __init__(self, t, v, i, clamp_mode, sampling_rate, sweep_number=None, epochs=None, test_pulse=True):
        self._t = t
        self._v = v
        self._i = i
        self.sampling_rate = sampling_rate
        self.sweep_number = sweep_number
        self.clamp_mode = clamp_mode

        if epochs:
            self.epochs = epochs
        else:
            self.epochs = {}

        self.selected_epoch_name = "sweep"

        if self.clamp_mode == "CurrentClamp":
            self.response = self._v
            self.stimulus = self._i
        else:
            self.response = self._i
            self.stimulus = self._v

        self.detect_epochs(test_pulse=test_pulse)

    @property
    def t(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._t[start_idx:end_idx+1]

    @property
    def v(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._v[start_idx:end_idx+1]

    @property
    def i(self):
        start_idx, end_idx = self.epochs[self.selected_epoch_name]
        return self._i[start_idx:end_idx+1]

    def select_epoch(self, epoch_name):
        """Select epoch by name if it exists. Return True if successful, False if not.
        """
        if epoch_name in self.epochs:
            self.selected_epoch_name = epoch_name
            return True
        else: 
            return False

    def set_time_zero_to_index(self, time_step):
        dt = 1. / self.sampling_rate
        self._t = self._t - time_step*dt

    def detect_epochs(self, test_pulse=False):
        """
        Detect epochs if they are not provided in the constructor

        """

        epoch_detectors = {
            "sweep": ep.get_sweep_epoch(self.response),
            "recording": ep.get_recording_epoch(self.response),
            "experiment": ep.get_experiment_epoch(self._i, self.sampling_rate),
            "test": ep.get_test_epoch(self._i),
            "stim": ep.get_stim_epoch(self._i, test_pulse=test_pulse),
        }

        for epoch_name, epoch_detector in epoch_detectors.items():
            if epoch_name not in self.epochs and epoch_detector[0] is not None:
                self.epochs[epoch_name] = epoch_detector


class SweepSet(object):
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def _prop(self, prop):
        return [getattr(s, prop) for s in self.sweeps]

    def select_epoch(self, epoch_name):
        for sweep in self.sweeps:
            sweep.select_epoch(epoch_name)

    def align_to_start_of_epoch(self, epoch_name):

        for sweep in self.sweeps:
            start_idx, end_idx = sweep.epochs[epoch_name]
            sweep.set_time_zero_to_index(start_idx)

    @property
    def t(self):
        return self._prop('t')

    @property
    def v(self):
        return self._prop('v')

    @property
    def i(self):
        return self._prop('i')

    @property
    def sweep_number(self):
        return self._prop('sweep_number')

    @property
    def sampling_rate(self):
        return self._prop('sampling_rate')
