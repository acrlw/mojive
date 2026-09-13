"""Compare merged ImGui uploads on one WebGPU device; this is not a full UI frame benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import wgpu
from imgui_bundle import imgui
from wgpu.utils.imgui import ImguiWgpuBackend

from mojive.ui.wgpu_buffers import ImguiBuffers


def draw_data(list_count, quads):
    data = imgui.ImDrawData()
    data.valid = True
    data.display_size = (1280, 720)
    data.framebuffer_scale = (1, 1)
    lists = []
    for index in range(list_count):
        draw_list = imgui.ImDrawList(imgui.get_draw_list_shared_data())
        draw_list._reset_for_new_frame()
        draw_list.push_clip_rect((0, 0), (1280, 720))
        for quad in range(quads // list_count + (index < quads % list_count)):
            x, y = quad % 100 * 12, quad // 100 * 12
            draw_list.add_rect_filled((x, y), (x + 8, y + 8), 0xFFFFFFFF)
        data.add_draw_list(draw_list)
        lists.append(draw_list)
    return data, lists


def run(device, frames, repeats, counts, quads):
    reference = ImguiWgpuBackend(device, "rgba8unorm")
    current = ImguiBuffers(device)
    fence = device.create_buffer(
        size=4, usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST
    )
    token = np.array([1], np.uint32)

    def drain():
        # A mapped read after both queue writes measures completion without the
        # installed wgpu-py's incompatible work-done callback binding.
        device.queue.write_buffer(fence, 0, token)
        assert bytes(device.queue.read_buffer(fence)) == token.tobytes()

    submit = {"upstream": reference._update_vertex_buffer, "reused": current.upload}
    raw, summaries = [], []
    try:
        for count in counts:
            data, lists = draw_data(count, quads)
            # Check byte equivalence outside timing, while still issuing the real uploads.
            writes = []
            original_write = device.queue.write_buffer

            def capture(
                buffer,
                offset,
                values,
                *args,
                writes=writes,
                original_write=original_write,
                **kwargs,
            ):
                writes.append(bytes(memoryview(values)))
                original_write(buffer, offset, values, *args, **kwargs)

            device.queue.write_buffer = capture
            try:
                submit["upstream"](data)
                expected = tuple(writes)
                writes.clear()
                submit["reused"](data)
                if tuple(writes) != expected:
                    raise RuntimeError("Paired ImGui upload bytes differ")
            finally:
                device.queue.write_buffer = original_write
            for upload in submit.values():
                for _ in range(10):
                    upload(data)
                drain()
            for repeat in range(repeats):
                order = ("upstream", "reused") if repeat % 2 == 0 else ("reused", "upstream")
                for mode in order:
                    drain()
                    allocations = current.allocations
                    start = perf_counter()
                    samples = []
                    for frame in range(frames):
                        lists[0].vtx_buffer[0].pos.x = frame % 16 * 0.125
                        tick = perf_counter()
                        submit[mode](data)
                        elapsed = (perf_counter() - tick) * 1000
                        samples.append(elapsed)
                        raw.append(
                            {
                                "lists": count,
                                "mode": mode,
                                "repeat": repeat,
                                "frame": frame,
                                "upload_ms": elapsed,
                            }
                        )
                    drain()
                    completed_ms = (perf_counter() - start) * 1000 / frames
                    summaries.append(
                        {
                            "lists": count,
                            "mode": mode,
                            "repeat": repeat,
                            "median_ms": float(np.median(samples)),
                            "p95_ms": float(np.percentile(samples, 95)),
                            "completed_ms_per_frame": completed_ms,
                            "new_buffer_allocations": current.allocations - allocations
                            if mode == "reused"
                            else None,
                            "upload_bytes_per_frame": sum(map(len, expected)),
                            "equal_upload_bytes": True,
                        }
                    )
            del data, lists
    finally:
        current.close()
        for buffer in (
            reference._vertex_buffer,
            reference._index_buffer,
            reference._uniform_buffer,
        ):
            if buffer is not None:
                buffer.destroy()
        fence.destroy()
    return raw, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lists", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--quads", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("output/ui-upload-benchmark"))
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.quads, *args.lists) <= 0 or max(args.lists) > args.quads:
        parser.error("Counts must be positive and lists must not exceed quads")
    args.output.mkdir(parents=True, exist_ok=True)
    context = imgui.create_context()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    try:
        raw, summaries = run(device, args.frames, args.repeats, args.lists, args.quads)
        with (args.output / "frames.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw[0])
            writer.writeheader()
            writer.writerows(raw)
        report = {
            "gpu": dict(adapter.info),
            "wgpu": version("wgpu"),
            "imgui_bundle": version("imgui-bundle"),
            "frames": args.frames,
            "repeats": args.repeats,
            "quads": args.quads,
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
                for name, value in (
                    ("upstream", ImguiWgpuBackend._update_vertex_buffer),
                    ("reused", ImguiBuffers),
                )
            },
            "cases": summaries,
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        device.destroy()
        imgui.destroy_context(context)


if __name__ == "__main__":
    main()
