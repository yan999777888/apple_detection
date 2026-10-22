"""Supervise one authorized refinement, full-validation or GPU-check job.

Example (use a new log name for every invocation):
  python scripts/watch_refinement.py --log-name refine800 -- scripts/train_refinement.py \
      --data data/papple/data --output runs/refine800 --initial-checkpoint runs/full01_local/best.pt

This supervisor never imports torch. All child arguments are passed as a list
without a shell. STOP_LOCAL_TRAINING stops only this supervisor's own child.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

from atomic_io import replace_with_retry


GIB = 1024 ** 3
PATH_FLAGS = {"--data", "--output", "--resume", "--initial-checkpoint"}
TRAIN_VALUE_FLAGS = {"--epochs", "--batch-size", "--accumulate", "--min-size", "--max-size", "--seed", "--warmup-updates"}
TRAIN_FLOAT_FLAGS = {"--head-lr", "--backbone-lr"}
TRAIN_BOOL_FLAGS = set()
V2_COCO_SHA256 = '73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e'
V2_COCO_FILENAME = 'maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
V2_SCRIPTS = {'train_v2_amodal.py', 'verify_v2_gpu.py'}
SCRIPT_CHOICE_FLAGS = {
    'train_guided_refinement.py': {'--guidance-mode': {'visible', 'uniform'}},
    'train_v2_amodal.py': {'--coco-sha256': {V2_COCO_SHA256}},
    'verify_v2_gpu.py': {'--coco-sha256': {V2_COCO_SHA256}, '--min-size': {'800', '1024'}},
}
SCRIPT_RULES = {
    'train_yolo_box.py': ({'--data', '--output', '--checkpoint'},
                          {'--epochs', '--patience', '--imgsz', '--batch-size', '--seed'}, set(), 'status.json'),
    'train_v2_amodal.py': ({'--data', '--output', '--resume', '--target-cache', '--coco-weights',
                            '--copy-paste-bank'},
                           TRAIN_VALUE_FLAGS | {'--copy-paste-count', '--expected-val-images'},
                           TRAIN_FLOAT_FLAGS | {'--copy-paste'}, 'status.json'),
    'verify_v2_gpu.py': ({'--data', '--output', '--coco-weights'}, set(), set(), 'verification.json'),
    'train_refinement.py': (PATH_FLAGS | {'--target-cache'}, TRAIN_VALUE_FLAGS, TRAIN_FLOAT_FLAGS, 'status.json'),
    'train_guided_refinement.py': (PATH_FLAGS | {'--target-cache'}, TRAIN_VALUE_FLAGS | {'--guidance-chunk-size'}, TRAIN_FLOAT_FLAGS, 'status.json'),
    'evaluate_checkpoint.py': ({'--data', '--output', '--checkpoint'}, set(), set(), 'status.json'),
    'verify_refinement_gpu.py': ({'--data', '--output', '--checkpoint'}, {'--min-size'}, set(), 'verification.json'),
    'diagnose_resume_drift.py': ({'--data', '--output', '--checkpoint'}, {'--min-size'}, set(), 'diagnosis.json'),
    'compare_box_nms.py': ({'--data', '--output', '--checkpoint'}, set(), set(), 'metadata.json'),
    'verify_visible_guidance_gpu.py': ({'--data', '--output', '--checkpoint'}, {'--min-size'}, set(), 'verification.json'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path, max_attempts=6)


def inside_root(value, root):
    candidate = Path(value).expanduser()
    candidate = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if candidate.drive.upper() != "F:" or not candidate.is_relative_to(root):
        raise ValueError(f"Path must remain inside the F: project: {value}")
    return candidate


def validate_command(arguments, root):
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        raise ValueError("Place the Python script and its arguments after --")
    script = inside_root(arguments[0], root)
    allowed = {root / 'scripts' / name for name in SCRIPT_RULES}
    if script not in {item.resolve() for item in allowed} or not script.is_file():
        raise ValueError('Only explicitly listed local refinement/check scripts may be supervised')
    path_flags, value_flags, float_flags, _status_name = SCRIPT_RULES[script.name]
    choice_flags = SCRIPT_CHOICE_FLAGS.get(script.name, {})
    bool_flags = TRAIN_BOOL_FLAGS
    result, paths, seen = [sys.executable, "-u", str(script)], {}, set()
    index = 1
    while index < len(arguments):
        token = arguments[index]
        flag, equal, attached = token.partition("=")
        if flag in seen:
            raise ValueError(f"Duplicate argument is not permitted: {flag}")
        seen.add(flag)
        if flag in bool_flags:
            if equal:
                raise ValueError(f"Flag does not take a value: {flag}")
            result.append(flag)
            index += 1
            continue
        if flag not in path_flags | value_flags | float_flags | choice_flags.keys():
            raise ValueError(f"Unsupported child argument: {flag}")
        if equal:
            value = attached
            index += 1
        else:
            if index + 1 >= len(arguments):
                raise ValueError(f"Missing value for {flag}")
            value = arguments[index + 1]
            index += 2
        if not value:
            raise ValueError(f"Empty value for {flag}")
        if flag in path_flags:
            value = str(inside_root(value, root))
            paths[flag] = value
        elif flag in float_flags:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"Learning rate must be finite and positive: {flag}")
        elif flag in choice_flags:
            if flag == '--coco-sha256':
                value = value.lower()
            if value not in choice_flags[flag]:
                raise ValueError(f"Unsupported choice for {flag}: {value}")
        else:
            int(value)  # No extra script names, expression strings or path values.
        result.extend([flag, value])
    if script.name == 'verify_v2_gpu.py':
        # Match the probe's real defaults, making them explicit for supervision.
        defaults = {'--data': str(root / 'data/papple/data'),
                    '--coco-weights': str(root / 'cache/torch/hub/checkpoints' / V2_COCO_FILENAME),
                    '--coco-sha256': V2_COCO_SHA256}
        for flag, value in defaults.items():
            if flag not in seen:
                if flag in path_flags:
                    paths[flag] = value = str(inside_root(value, root))
                result.extend([flag, value])
                seen.add(flag)
    if script.name in V2_SCRIPTS:
        if '--coco-sha256' not in seen or '--coco-weights' not in paths:
            raise ValueError('V2 requires local --coco-weights and complete --coco-sha256')
        source = Path(paths['--coco-weights'])
        if source.name != V2_COCO_FILENAME or not source.is_file():
            raise ValueError('Use the existing officially named local V2 COCO weights file')
        source_flag = '--coco-weights'
    else:
        source_flag = '--initial-checkpoint' if script.name in {'train_refinement.py', 'train_guided_refinement.py'} else '--checkpoint'
    if not {'--output', '--data', source_flag} <= paths.keys():
        raise ValueError(f'--output, --data and {source_flag} are required')
    return result, script, paths


def driver_sample():
    query = "index,name,temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if completed.returncode:
            return {"error": completed.stderr.strip()[:500]}
        cells = next(csv.reader(completed.stdout.strip().splitlines()))
        keys = ["index", "name", "temperature_c", "used_mib", "total_mib", "utilization_percent", "power_w"]
        result = {}
        for key, value in zip(keys, cells):
            if key == "name":
                result[key] = value.strip()
                continue
            try:
                number = float(value.strip())
                result[key] = number if math.isfinite(number) else None
            except ValueError:
                result[key] = None
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def identity(psutil, pid):
    process = psutil.Process(pid)
    return {"pid": process.pid, "create_time": process.create_time()}


def identity_alive(psutil, recorded):
    if not recorded or recorded.get("pid") is None or recorded.get("create_time") is None:
        return False
    try:
        process = psutil.Process(int(recorded["pid"]))
        return process.is_running() and abs(process.create_time() - float(recorded["create_time"])) < 0.01
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        # Unreadable live ownership cannot safely be classified as stale.
        return True


class SupervisorLock:
    """Windows byte lock plus recorded PID/create-time for stale/orphan audit."""
    def __init__(self, root, psutil):
        self.path = root / "reports" / "supervisor.lock"
        self.psutil = psutil
        self.handle = None
        self.payload = None

    def __enter__(self):
        import msvcrt
        try:
            self.handle = self.path.open("x+b")
        except FileExistsError:
            self.handle = self.path.open("r+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Another supervisor holds {self.path}; no child was started") from exc
        try:
            self.handle.seek(0)
            previous_text = self.handle.read().decode("utf-8").strip()
            previous = json.loads(previous_text) if previous_text else {}
            if previous.get("phase") != "released":
                for key in ("supervisor", "child"):
                    if identity_alive(self.psutil, previous.get(key)):
                        raise RuntimeError(f"Recorded {key} is still alive; refusing a second GPU job: {previous.get(key)}")
            self.payload = {"host": platform.node(), "phase": "acquired", "supervisor": identity(self.psutil, os.getpid()),
                            "child": None, "acquired_utc": now(), "previous_owner": previous.get("supervisor")}
            self.update()
            return self
        except BaseException:
            self._unlock()
            raise

    def update(self, **fields):
        self.payload.update(fields, updated_utc=now())
        encoded = json.dumps(self.payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.handle.seek(0)
        self.handle.write(encoded)
        self.handle.truncate()
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def _unlock(self):
        if self.handle is not None:
            import msvcrt
            try:
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                self.handle.close()
                self.handle = None

    def __exit__(self, exc_type, exc, tb):
        try:
            # A failed termination must leave the child's identity marked as
            # active, so a future supervisor cannot start alongside an orphan.
            child_alive = identity_alive(self.psutil, self.payload.get("child"))
            self.update(phase="orphan_child_alive" if child_alive else "released", released_utc=now())
        finally:
            self._unlock()


def stop_owned_child(child, child_identity, psutil):
    if child is None or child.poll() is not None:
        return "already_exited"
    if not child_identity or not identity_alive(psutil, child_identity):
        return "not_terminated_identity_mismatch"
    # Popen retains the original Windows process handle. Only this exact child
    # is terminated: no PID-name sweeps, descendants or unrelated GPU processes.
    child.terminate()
    try:
        child.wait(timeout=15)
        return "terminated_owned_child"
    except subprocess.TimeoutExpired:
        if not identity_alive(psutil, child_identity):
            return "not_killed_identity_mismatch"
        child.kill()
        child.wait(timeout=15)
        return "killed_owned_child_after_timeout"


def read_tail(path, maximum_bytes=24000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            length = stream.tell()
            stream.seek(max(0, length - maximum_bytes))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def run(args, root, psutil):
    command, script, paths = validate_command(args.command, root)
    wall_limit = getattr(args, 'wall_time_seconds', None)
    if wall_limit is not None and (isinstance(wall_limit, bool) or not isinstance(wall_limit, int) or not 1 <= wall_limit <= 8 * 3600):
        raise ValueError('Optional wall-time limit must be an integer from 1 to 28800 seconds')
    name = args.log_name
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name) or name.endswith(".") or name.split(".")[0].upper() in reserved:
        raise ValueError("Use a simple, unique --log-name of ASCII letters, digits, underscore, dot or hyphen")
    stdout_path = root / "logs" / f"{name}.out.log"
    stderr_path = root / "logs" / f"{name}.err.log"
    telemetry_path = root / "logs" / f"{name}.monitor.jsonl"
    summary_path = root / "reports" / f"{name}_supervisor.json"
    current_path = root / "reports" / "current_job.json"
    for path in (stdout_path, stderr_path, telemetry_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Log name already used; choose a new name: {path}")
    runner_status = Path(paths["--output"]) / SCRIPT_RULES[script.name][3]
    stop_file = root / "STOP_LOCAL_TRAINING"
    with SupervisorLock(root, psutil) as lock:
        report = {
            "log_name": name, "phase": "starting", "started_utc": now(), "host": platform.node(),
            "supervisor": identity(psutil, os.getpid()), "child": None, "command": command,
            "runner_status_path": str(runner_status), "output": paths["--output"],
            "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "telemetry_path": str(telemetry_path),
            "summary_path": str(summary_path), "stop_file": str(stop_file), "exit_code": None,
            "limits": {"temperature_c": 84, "minimum_available_ram_bytes": 2 * GIB,
                       "consecutive_missing_telemetry": 3, "sample_interval_seconds": 5, "wall_time_limit": wall_limit},
            "monitor_summary": {"samples": 0, "maximum_temperature_c": None, "minimum_available_ram_bytes": None,
                                "maximum_driver_memory_used_mib": None, "maximum_child_rss_bytes": None},
            "accuracy_note": "Resource monitoring does not establish model quality; inspect the runner's validation results.",
        }
        child, child_identity = None, None
        started = time.monotonic()
        stopped = None
        missing_telemetry = 0
        stdout = stderr = telemetry = None

        def save():
            report["updated_utc"] = now()
            report["elapsed_seconds"] = time.monotonic() - started
            atomic_json(summary_path, report)
            atomic_json(current_path, report)

        def take_sample():
            sample = {"utc": now(), "elapsed_seconds": time.monotonic() - started, "gpu": driver_sample(),
                      "available_ram_bytes": int(psutil.virtual_memory().available), "child_rss_bytes": None}
            if child_identity and identity_alive(psutil, child_identity):
                try:
                    sample["child_rss_bytes"] = psutil.Process(child_identity["pid"]).memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            summary = report["monitor_summary"]
            summary["samples"] += 1
            for source, destination in ((sample["gpu"].get("temperature_c"), "maximum_temperature_c"),
                                        (sample["gpu"].get("used_mib"), "maximum_driver_memory_used_mib"),
                                        (sample["child_rss_bytes"], "maximum_child_rss_bytes")):
                if source is not None:
                    summary[destination] = source if summary[destination] is None else max(summary[destination], source)
            available = sample["available_ram_bytes"]
            summary["minimum_available_ram_bytes"] = available if summary["minimum_available_ram_bytes"] is None else min(summary["minimum_available_ram_bytes"], available)
            telemetry.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
            telemetry.flush()
            report["latest_sample"] = sample
            return sample

        def resource_reason(sample):
            nonlocal missing_telemetry
            if wall_limit is not None and time.monotonic() - started >= wall_limit:
                return 'wall_time_limit'
            gpu = sample["gpu"]
            temperature = gpu.get("temperature_c")
            telemetry_valid = temperature is not None and gpu.get("used_mib") is not None and gpu.get("total_mib") is not None
            missing_telemetry = 0 if telemetry_valid else missing_telemetry + 1
            if temperature is not None and temperature >= 84:
                return "temperature_limit"
            if sample["available_ram_bytes"] < 2 * GIB:
                return "available_ram_limit"
            if missing_telemetry >= 3:
                return "gpu_telemetry_unavailable"
            return None

        try:
            stdout = stdout_path.open("x", encoding="utf-8")
            stderr = stderr_path.open("x", encoding="utf-8")
            telemetry = telemetry_path.open("x", encoding="utf-8")
            save()
            if stop_file.exists():
                stopped = "stop_file_present_before_start"
            # Require complete counters before launching. Three failed attempts
            # are logged; no GPU work is started blindly during that interval.
            while not stopped:
                initial = take_sample()
                stopped = resource_reason(initial)
                if stopped or missing_telemetry == 0:
                    break
                save()
                time.sleep(5)
                if stop_file.exists():
                    stopped = "stop_file_present_before_start"
            if stopped:
                report["phase"] = "resource_stopped"
            else:
                child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                try:
                    child_identity = identity(psutil, child.pid)
                except psutil.NoSuchProcess:
                    child.wait(timeout=10)
                report["child"] = child_identity or {"pid": child.pid, "create_time": None}
                lock.update(phase="running", child=child_identity, log_name=name, summary_path=str(summary_path))
                report["phase"] = "running"
                save()
                print(json.dumps({"phase": "running", "child": report["child"], "current_job": str(current_path)}, ensure_ascii=False), flush=True)
                next_sample = time.monotonic()
                while child.poll() is None:
                    if stop_file.exists():
                        stopped = "stop_file_requested"
                        break
                    current = time.monotonic()
                    if current >= next_sample:
                        sample = take_sample()
                        stopped = resource_reason(sample)
                        save()
                        if stopped:
                            break
                        next_sample = current + 5
                    try:
                        child.wait(timeout=min(1.0, max(0.05, next_sample - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        pass
                if stopped:
                    report["phase"] = "resource_stopped"
                else:
                    report["exit_code"] = child.returncode
                    report["phase"] = "complete" if child.returncode == 0 else "failed"
        except KeyboardInterrupt:
            stopped = "supervisor_keyboard_interrupt"
            report["phase"] = "resource_stopped"
        except BaseException as exc:
            report["phase"] = "failed"
            report["supervisor_error"] = f"{type(exc).__name__}: {exc}"
            report["supervisor_traceback"] = traceback.format_exc()
        finally:
            try:
                if child is not None and child.poll() is None:
                    report["termination_action"] = stop_owned_child(child, child_identity, psutil)
                if child is not None:
                    report["exit_code"] = child.poll()
                if stopped:
                    report["stop_reason"] = stopped
                for stream in (stdout, stderr, telemetry):
                    if stream is not None:
                        stream.close()
                failure_tail = read_tail(stderr_path) + "\n" + read_tail(stdout_path)
                report["oom_reported_in_logs"] = bool(re.search(r"CUDA out of memory|OutOfMemoryError|CUDA error: out of memory", failure_tail, re.IGNORECASE))
                if report["oom_reported_in_logs"] and report["phase"] == "complete":
                    report["phase"] = "failed"
                    report["failure_reason"] = "oom_in_logs_despite_zero_exit"
                report["automatic_retry_or_recipe_change"] = False
                report["finished_utc"] = now()
                save()
            except BaseException as exc:
                report["phase"] = "failed"
                report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                report["finished_utc"] = now()
                save()
        print(json.dumps({"phase": report["phase"], "exit_code": report["exit_code"], "stop_reason": report.get("stop_reason"),
                          "summary": str(summary_path)}, ensure_ascii=False), flush=True)
        return 0 if report["phase"] == "complete" else 2 if report["phase"] == "resource_stopped" else 1


def main():
    from runtime import configure_paths
    root = Path(configure_paths()).resolve()
    if platform.node().upper() != "YAN" or root.drive.upper() != "F:":
        raise RuntimeError("This supervisor is authorized only on YAN with an F: project")
    if Path(sys.executable).resolve().drive.upper() != "F:":
        raise RuntimeError("Use the project's Python environment on F:")
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-name", required=True)
    parser.add_argument('--wall-time-seconds', type=int, help='Optional maximum duration, including initial resource checks')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return run(args, root, psutil)


if __name__ == "__main__":
    raise SystemExit(main())
