import json
import math
import unittest

from apple_research_audit.profiling import PipelineProfiler


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class ProfilingTests(unittest.TestCase):
    def test_idle_queue_and_publication_are_in_wall_fps(self):
        profiler = PipelineProfiler()
        a = profiler.receive("a", received_at_s=0, captured_at_s=-.1,
                             capture_provenance="camera timestamp mapped to perf_counter")
        a.start(.2)
        a.result(.3)
        a.publish(.4)
        b = profiler.receive("b", received_at_s=.8, captured_at_s=.7,
                             capture_provenance="same mapping")
        b.start(.9)
        b.result(1.0)
        b.publish(1.1)
        summary = profiler.summary()
        self.assertAlmostEqual(summary["wall_throughput_fps"], 2/1.1)
        self.assertAlmostEqual(summary["active_service_fps"], 10)
        latency = summary["delivered_latency"]
        self.assertAlmostEqual(latency["queue"]["mean_ms"], 150)
        self.assertAlmostEqual(latency["publication"]["mean_ms"], 100)
        self.assertAlmostEqual(latency["capture_to_publish"]["p50_ms"], 450)
        self.assertAlmostEqual(latency["capture_to_publish"]["p95_ms"], 495)
        self.assertAlmostEqual(profiler.summary(observation_end_s=2)["wall_throughput_fps"], 1)

    def test_drop_not_counted_as_delivery_or_in_delivered_latency(self):
        profiler = PipelineProfiler()
        a = profiler.receive("ready", received_at_s=0)
        a.start(0)
        a.result(.1)
        a.publish(.2)
        b = profiler.receive("stale", received_at_s=.3)
        b.start(.4)
        b.result(.6)
        b.drop("superseded result", .7)
        c = profiler.receive("queued", received_at_s=.8)
        c.drop("queue overflow", 1)
        summary = profiler.summary()
        self.assertEqual(summary["delivered_frames"], 1)
        self.assertEqual(summary["dropped_frames"], 2)
        self.assertEqual(summary["result_frames"], 2)
        self.assertEqual(summary["wall_throughput_fps"], 1)
        self.assertAlmostEqual(summary["active_service_fps"], 2/.3)
        self.assertEqual(summary["delivered_latency"]["service"]["samples"], 1)

    def test_unknown_capture_is_not_zero_or_receive_timestamp(self):
        profiler = PipelineProfiler()
        frame = profiler.receive("a", received_at_s=0)
        frame.start(0)
        frame.result(.1)
        frame.publish(.2)
        self.assertIsNone(frame.to_dict()["frame_age_ms"])
        latency = profiler.summary()["delivered_latency"]
        self.assertIsNone(latency["capture_to_publish"])
        self.assertEqual(latency["unknown_capture_frames"], 1)
        self.assertAlmostEqual(latency["receive_to_publish"]["mean_ms"], 200)

    def test_sync_stage_wait_is_included_and_scope_is_recorded(self):
        clock = Clock()
        profiler = PipelineProfiler(clock=clock, clock_provenance="test monotonic clock")
        frame = profiler.receive("a")
        frame.start()
        calls = []

        def sync():
            calls.append(clock.now)
            clock.now += .05

        with frame.stage("gpu", synchronize=sync):
            clock.now += .01
        with frame.stage("host"):
            clock.now += .02
        frame.result()
        frame.publish()
        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(frame.stages[0]["duration_ms"], 60)
        self.assertEqual(frame.stages[0]["timing_scope"], "device_synchronized_wall")
        self.assertEqual(frame.stages[1]["timing_scope"], "host_wall_no_device_sync")
        self.assertAlmostEqual(profiler.summary()["delivered_latency"]["service"]["mean_ms"], 130)
        json.dumps({"frames": [frame.to_dict()], "summary": profiler.summary()}, allow_nan=False)

    def test_timestamp_and_lifecycle_rejections(self):
        profiler = PipelineProfiler()
        for bad in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                profiler.receive("bad", received_at_s=bad)
        with self.assertRaises(ValueError):
            profiler.receive("bad", received_at_s=0, captured_at_s=-1)
        with self.assertRaises(ValueError):
            profiler.receive("bad", received_at_s=0, captured_at_s=1, capture_provenance="mapped")
        frame = profiler.receive("a", received_at_s=1)
        with self.assertRaises(ValueError):
            frame.start(.9)
        with self.assertRaises(ValueError):
            frame.publish(2)
        with self.assertRaises(ValueError):
            profiler.summary()
        frame.start(2)
        with self.assertRaises(ValueError):
            frame.result(1.5)
        frame.result(3)
        frame.publish(4)
        with self.assertRaises(ValueError):
            frame.drop("late", 5)
        with self.assertRaises(ValueError):
            profiler.receive("a", received_at_s=5)
        with self.assertRaises(ValueError):
            profiler.summary(observation_end_s=3)
        with self.assertRaises(ValueError):
            profiler.summary(observation_start_s=2)

    def test_exception_stage_is_recorded_as_failed(self):
        clock = Clock()
        profiler = PipelineProfiler(clock=clock)
        frame = profiler.receive("a")
        frame.start()
        with self.assertRaises(RuntimeError):
            with frame.stage("inference"):
                clock.now = .1
                raise RuntimeError("model failed")
        frame.drop("inference failed")
        self.assertFalse(frame.stages[0]["success"])
        self.assertEqual(profiler.summary()["delivered_frames"], 0)

    def test_empty_observation_has_no_invented_fps(self):
        profiler = PipelineProfiler()
        self.assertIsNone(profiler.summary()["wall_throughput_fps"])
        self.assertEqual(profiler.summary(observation_start_s=0, observation_end_s=1)["wall_throughput_fps"], 0)

    def test_result_cannot_precede_stage_end(self):
        clock = Clock()
        profiler = PipelineProfiler(clock=clock)
        frame = profiler.receive("a")
        frame.start()
        with frame.stage("work"):
            clock.now = .5
        with self.assertRaises(ValueError):
            frame.result(.4)
        frame.result()
        frame.publish()

    def test_result_synchronization_precedes_result_timestamp(self):
        clock = Clock()
        frame = PipelineProfiler(clock=clock).receive("a")
        frame.start()

        def sync():
            clock.now = .4

        with self.assertRaises(ValueError):
            frame.result(.2, synchronize=sync)
        self.assertEqual(clock.now, 0)
        frame.result(synchronize=sync)
        self.assertEqual(frame.result_at_s, .4)
        self.assertEqual(frame.result_readiness, "device_synchronized_at_result")

    def test_mixed_capture_coverage_is_not_reported_as_complete_e2e(self):
        profiler = PipelineProfiler()
        a = profiler.receive("known", received_at_s=0, captured_at_s=0,
                             capture_provenance="mapped")
        a.start(0)
        a.result(.1)
        a.publish(.2)
        b = profiler.receive("unknown", received_at_s=.3)
        b.start(.3)
        b.result(.4)
        b.publish(.5)
        latency = profiler.summary()["delivered_latency"]
        self.assertIsNone(latency["capture_to_publish"])
        self.assertEqual(latency["known_capture_to_publish_subset"]["samples"], 1)
        self.assertEqual(latency["unknown_capture_frames"], 1)

    def test_failed_stage_cannot_leave_frame_ready_while_open(self):
        clock = Clock()
        frame = PipelineProfiler(clock=clock).receive("a")
        frame.start()
        with frame.stage("work"):
            with self.assertRaises(ValueError):
                frame.result()
            with self.assertRaises(ValueError):
                frame.drop("too soon")
        frame.drop("finished")


if __name__ == "__main__":
    unittest.main()
