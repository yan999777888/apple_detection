"""Dependency-free timing for a synchronous vision loop; no model code is changed.

Use ONE monotonic clock domain. Map camera timestamps into that domain before
passing captured_at_s and describe the mapping in capture_provenance. If that
mapping is unavailable, omit capture time: camera-to-output latency is unknown.

    profiler = PipelineProfiler()
    frame = profiler.receive("42", capture_provenance=None)
    frame.start()
    with frame.stage("segmentation", synchronize=torch.cuda.synchronize):
        result = model(image)
    frame.result()  # all required GPU work must be complete at this boundary
    publish(result)
    frame.publish()  # after publication, not when publication is scheduled
    report = profiler.summary()

Stage measurements are diagnostic and may overlap; their means are NEVER summed
to compute throughput. Without an injected synchronization function a GPU stage
measures host wall time only. No camera, model, files, or GPU is accessed here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Callable, Iterator


def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Timestamps must be finite monotonic-clock seconds")
    return value


def _stats_ms(seconds: list[float]) -> dict | None:
    if not seconds:
        return None
    values = sorted(value * 1000 for value in seconds)

    def percentile(q: float) -> float:
        index = (len(values) - 1) * q
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {"samples": len(values), "mean_ms": sum(values) / len(values),
            "p50_ms": percentile(.50), "p95_ms": percentile(.95),
            "min_ms": values[0], "max_ms": values[-1]}


@dataclass
class FrameTiming:
    frame_id: str
    received_at_s: float
    captured_at_s: float | None
    clock_provenance: str
    capture_provenance: str | None
    started_at_s: float | None = None
    result_at_s: float | None = None
    published_at_s: float | None = None
    dropped_at_s: float | None = None
    drop_reason: str | None = None
    result_readiness: str | None = None
    stages: list[dict] = field(default_factory=list)
    _clock: Callable[[], float] = field(default=perf_counter, repr=False)
    _stage_open: int = field(default=0, repr=False)
    _last_event_s: float | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        return self.published_at_s is not None or self.dropped_at_s is not None

    def _next_time(self, at_s: float | None) -> float:
        if self.terminal:
            raise ValueError("Frame is already published or dropped")
        value = _time(self._clock() if at_s is None else at_s)
        previous = max(item for item in (self._last_event_s, self.result_at_s,
                                         self.started_at_s, self.received_at_s)
                       if item is not None)
        if value < previous:
            raise ValueError("Frame timestamps must be chronological")
        self._last_event_s = value
        return value

    def start(self, at_s: float | None = None) -> None:
        if self.started_at_s is not None:
            raise ValueError("Frame processing has already started")
        self.started_at_s = self._next_time(at_s)

    def result(self, at_s: float | None = None,
               synchronize: Callable[[], None] | None = None) -> None:
        """Mark a ready result; synchronize outstanding device work if necessary."""
        if self.started_at_s is None or self.result_at_s is not None or self._stage_open:
            raise ValueError("Result requires a started frame with all stages closed")
        if synchronize is not None:
            if at_s is not None:
                raise ValueError("Read the clock after synchronization; do not supply at_s")
            synchronize()
        self.result_at_s = self._next_time(at_s)
        self.result_readiness = ("device_synchronized_at_result" if synchronize
                                 else "caller_asserted_result_ready")

    def publish(self, at_s: float | None = None) -> None:
        if self.result_at_s is None:
            raise ValueError("Publish requires a completed result")
        self.published_at_s = self._next_time(at_s)

    def drop(self, reason: str, at_s: float | None = None) -> None:
        if not reason.strip() or self._stage_open:
            raise ValueError("Drop requires a reason and all stages closed")
        self.dropped_at_s = self._next_time(at_s)
        self.drop_reason = reason

    @contextmanager
    def stage(self, name: str,
              synchronize: Callable[[], None] | None = None) -> Iterator[None]:
        """Record a stage, including completion synchronization in its duration.

        Pre-stage synchronization occurs before its timer to exclude earlier GPU
        work. It remains inside the enclosing frame's service time.
        """
        if (not name.strip() or self.started_at_s is None or self.result_at_s is not None
                or self.terminal):
            raise ValueError("Named stages require a started, unfinished frame")
        if synchronize is not None:
            synchronize()
        start = self._next_time(None)
        self._stage_open += 1
        success = False
        try:
            yield
            success = True
        finally:
            self._stage_open -= 1
            if synchronize is not None:
                synchronize()
            end = self._next_time(None)
            if end < start:
                raise ValueError("Stage clock moved backwards")
            self.stages.append({"name": name, "started_at_s": start,
                                "ended_at_s": end, "duration_ms": (end-start)*1000,
                                "success": success,
                                "timing_scope": "device_synchronized_wall" if synchronize
                                else "host_wall_no_device_sync"})

    def to_dict(self) -> dict:
        # Do not serialize the callback: injected clocks may contain locks/state.
        values = {key: value for key, value in self.__dict__.items() if not key.startswith("_")}
        values["frame_age_ms"] = (
            (self.published_at_s-self.captured_at_s)*1000
            if self.published_at_s is not None and self.captured_at_s is not None else None)
        return values


class PipelineProfiler:
    def __init__(self, clock: Callable[[], float] = perf_counter,
                 clock_provenance: str = "Python time.perf_counter monotonic seconds"):
        if not clock_provenance.strip():
            raise ValueError("Clock provenance is required")
        self.clock = clock
        self.clock_provenance = clock_provenance
        self.frames: dict[str, FrameTiming] = {}

    def receive(self, frame_id: str, *, captured_at_s: float | None = None,
                received_at_s: float | None = None,
                capture_provenance: str | None = None) -> FrameTiming:
        if not isinstance(frame_id, str) or not frame_id or frame_id in self.frames:
            raise ValueError("Frame IDs must be unique nonempty strings")
        received = _time(self.clock() if received_at_s is None else received_at_s)
        captured = None if captured_at_s is None else _time(captured_at_s)
        if captured is not None:
            if not capture_provenance or not capture_provenance.strip():
                raise ValueError("Describe how capture time maps into this monotonic clock")
            if captured > received:
                raise ValueError("Capture time cannot follow reception")
        frame = FrameTiming(frame_id, received, captured, self.clock_provenance,
                            capture_provenance, _clock=self.clock)
        self.frames[frame_id] = frame
        return frame

    def summary(self, *, observation_start_s: float | None = None,
                observation_end_s: float | None = None) -> dict:
        """Summarize terminal frames in an explicitly bounded observation window.

        The default window is first reception through last publish/drop. A caller
        can include longer camera wait/idle periods with explicit boundaries.
        Throughput counts actual published frames only. Service capacity counts
        all ready results (including those later dropped), and is not E2E FPS.
        """
        frames = list(self.frames.values())
        if any(not frame.terminal for frame in frames):
            raise ValueError("Publish or explicitly drop every received frame before summary")
        terminal_times = [frame.published_at_s if frame.published_at_s is not None
                          else frame.dropped_at_s for frame in frames]
        first = min((frame.received_at_s for frame in frames), default=None)
        last = max(terminal_times, default=None)
        start = first if observation_start_s is None else _time(observation_start_s)
        end = last if observation_end_s is None else _time(observation_end_s)
        if (start is None) != (end is None):
            raise ValueError("Empty observations require both window boundaries or neither")
        if start is not None and (end < start or (first is not None and start > first)
                                  or (last is not None and end < last)):
            raise ValueError("Observation window must enclose every receive and terminal event")
        elapsed = None if start is None else end-start
        delivered = [frame for frame in frames if frame.published_at_s is not None]
        ready = [frame for frame in frames if frame.result_at_s is not None]
        service_total = sum(frame.result_at_s-frame.started_at_s for frame in ready)
        ages = [frame.published_at_s-frame.captured_at_s for frame in delivered
                if frame.captured_at_s is not None]
        stage_groups: dict[tuple[str, str], list[float]] = {}
        for frame in frames:
            for stage in frame.stages:
                key = (stage["name"], stage["timing_scope"])
                stage_groups.setdefault(key, []).append(stage["duration_ms"]/1000)
        return {
            "schema": "apple_pipeline_timing_v1", "clock_provenance": self.clock_provenance,
            "received_frames": len(frames), "result_frames": len(ready),
            "delivered_frames": len(delivered), "dropped_frames": len(frames)-len(delivered),
            "observation_start_s": start, "observation_end_s": end,
            "observation_elapsed_s": elapsed,
            "wall_throughput_fps": len(delivered)/elapsed if elapsed and elapsed > 0 else None,
            "active_service_fps": len(ready)/service_total if service_total > 0 else None,
            "active_service_scope": "all result-ready frames, including subsequently dropped frames",
            "delivered_latency": {
                "queue": _stats_ms([f.started_at_s-f.received_at_s for f in delivered]),
                "service": _stats_ms([f.result_at_s-f.started_at_s for f in delivered]),
                "publication": _stats_ms([f.published_at_s-f.result_at_s for f in delivered]),
                "receive_to_publish": _stats_ms([f.published_at_s-f.received_at_s for f in delivered]),
                "capture_to_publish": _stats_ms(ages) if len(ages) == len(delivered) else None,
                "known_capture_to_publish_subset": _stats_ms(ages),
                "unknown_capture_frames": len(delivered)-len(ages),
            },
            "stages": [{"name": name, "timing_scope": scope, **_stats_ms(values)}
                       for (name, scope), values in stage_groups.items()],
            "stage_durations_used_to_compute_fps": False,
        }
