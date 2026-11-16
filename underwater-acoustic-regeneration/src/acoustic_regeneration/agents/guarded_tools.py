"""Feasibility gate for virtual settings; never supplies a cue label or expected answer."""
from .benchmark_tools import BenchmarkTools, qualifies


class GuardedBenchmarkTools(BenchmarkTools):
    def configure_display(self, window_ms):
        if isinstance(window_ms,bool) or window_ms not in self.analyzed:
            raise ValueError('Analyze a supported window before configuring it')
        measurement=self.analyze_window(window_ms)
        if not qualifies(measurement,self.task['constraints']):
            raise ValueError('Configuration rejected: the measured setting fails the requested bin-spacing, time-support, or observed-frame limits. Display unchanged. Choose a feasible measured setting or decline.')
        return super().configure_display(window_ms)
