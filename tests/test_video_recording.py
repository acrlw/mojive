"""Video contracts and real FFmpeg round trips independent of a graphics context."""

from __future__ import annotations

import builtins
import contextlib
import io
import subprocess
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

from mojive import VideoRecorder
from mojive.capture import recording as recording_module


@pytest.mark.parametrize("size", ((0, 10), (10, -1), (10, 20, 30)))
def test_video_rejects_invalid_size_without_creating_output(tmp_path, size):
    path = tmp_path / "new" / "bad.mp4"
    with pytest.raises(ValueError):
        VideoRecorder(path, size)
    assert not path.parent.exists()


@pytest.mark.parametrize("fps", (0, -1, float("nan"), float("inf")))
def test_video_rejects_invalid_fps(tmp_path, fps):
    with pytest.raises(ValueError, match="finite and positive"):
        VideoRecorder(tmp_path / "bad.mp4", (32, 24), fps=fps)


def test_video_rejects_unknown_pixel_format(tmp_path):
    with pytest.raises(ValueError, match="pixel_format"):
        VideoRecorder(tmp_path / "bad.mp4", (32, 24), pixel_format="typo")


@pytest.mark.parametrize(
    "options",
    (
        {"crf": -1},
        {"crf": 52},
        {"crf": 1.5},
        {"crf": float("nan")},
        {"bitrate": 0},
        {"bitrate": -1},
        {"bitrate": 2.5},
        {"crf": 20, "bitrate": 1_000_000},
        {"preset": "unknown"},
        {"codec": "msmpeg4", "crf": 20},
        {"codec": "msmpeg4", "preset": "fast"},
    ),
)
def test_invalid_encoding_settings_fail_before_creating_output(tmp_path, options):
    output = tmp_path / "uncreated/video.mp4"
    with pytest.raises(ValueError):
        VideoRecorder(output, (64, 48), **options)
    assert not output.parent.exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "options", ({}, {"crf": 18, "preset": "slow"}, {"bitrate": 2_000_000, "preset": "fast"})
)
def test_video_encoding_modes_reach_ffmpeg_and_produce_playable_video(tmp_path, options):
    path = tmp_path / "configured.mp4"
    with VideoRecorder(path, (64, 48), fps=12, pixel_format="yuv444p", **options) as video:
        args = video._process.args
        assert args[args.index("-preset") + 1] == options.get("preset", "medium")
        if "bitrate" in options:
            assert "-crf" not in args
            assert args[args.index("-b:v") + 1] == str(options["bitrate"])
        else:
            assert "-b:v" not in args
            assert args[args.index("-crf") + 1] == str(options.get("crf", 25))
        for value in (40, 100, 180):
            video.append(np.full((48, 64, 3), value, np.uint8))
    with contextlib.closing(imageio_ffmpeg.read_frames(str(path))) as reader:
        metadata = next(reader)
        assert metadata["size"] == (64, 48) and metadata["fps"] == 12
        frames = [np.frombuffer(frame, np.uint8) for frame in reader]
    assert len(frames) == 3
    assert [frame.mean() for frame in frames] == pytest.approx((40, 100, 180), abs=3)


@pytest.mark.integration
def test_lower_crf_preserves_more_detail_in_the_encoded_video(tmp_path):
    frame = np.random.default_rng(12).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    errors, sizes = [], []
    for crf in (12, 38):
        path = tmp_path / f"quality-{crf}.mp4"
        with VideoRecorder(
            path, (128, 96), crf=crf, preset="fast", pixel_format="yuv444p"
        ) as video:
            for _ in range(6):
                video.append(frame)
        with contextlib.closing(imageio_ffmpeg.read_frames(str(path))) as reader:
            next(reader)
            decoded = np.frombuffer(next(reader), np.uint8).reshape(frame.shape)
        errors.append(np.mean((decoded.astype(float) - frame.astype(float)) ** 2))
        sizes.append(path.stat().st_size)
    assert errors[0] < errors[1] * 0.5
    assert sizes[0] > sizes[1]


def test_video_timeout_kills_and_reaps_the_encoder(tmp_path, monkeypatch):
    process = SimpleNamespace(stdin=io.BytesIO(), returncode=None, waits=[], kills=0)

    def wait(timeout=None):
        process.waits.append(timeout)
        if timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        process.returncode = -9
        return -9

    def kill():
        process.kills += 1

    process.wait = wait
    process.kill = kill
    process.poll = lambda: process.returncode
    monkeypatch.setattr(recording_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")
    video = VideoRecorder(tmp_path / "slow.mp4", (32, 24))
    video.append(np.zeros((24, 32, 3), np.uint8))
    with pytest.raises(RuntimeError, match="did not finish within 30 seconds"):
        video.close()
    assert process.waits == [30.0, None]
    assert process.kills == 1
    assert process.stdin.closed
    assert video._stderr.closed
    video.close()


def test_missing_package_reports_the_install_command(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "imageio_ffmpeg":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(RuntimeError, match="pip install imageio-ffmpeg"):
        VideoRecorder(tmp_path / "bad.mp4", (32, 24))


def test_missing_ffmpeg_reports_the_executable_override(tmp_path, monkeypatch):
    def unavailable():
        raise RuntimeError("No ffmpeg exe could be found")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", unavailable)
    with pytest.raises(RuntimeError, match="IMAGEIO_FFMPEG_EXE"):
        VideoRecorder(tmp_path / "bad.mp4", (32, 24))


@pytest.mark.integration
def test_invalid_executable_is_reported_at_start(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", str(tmp_path / "missing-ffmpeg"))
    with pytest.raises(RuntimeError, match="Cannot start FFmpeg"):
        VideoRecorder(tmp_path / "bad.mp4", (32, 24))


@pytest.mark.integration
@pytest.mark.parametrize("pixel_format", ("yuv420p", "yuv444p"))
@pytest.mark.parametrize("size", ((64, 48), (63, 47), (64, 47), (63, 48)))
def test_video_round_trip_preserves_fps_frame_order_and_unscaled_pixels(
    tmp_path, pixel_format, size
):
    width, height = size
    padded = pixel_format == "yuv420p" and (width % 2 or height % 2)
    warning = (
        pytest.warns(RuntimeWarning, match="edge-padded") if padded else contextlib.nullcontext()
    )
    path = tmp_path / "rollout.mp4"
    frames = []
    with warning, VideoRecorder(path, size, fps=12.5, pixel_format=pixel_format) as video:
        for value in (40, 110, 180):
            # Strided RGBA input checks conversion without modifying caller-owned data.
            storage = np.full((height, width * 2, 4), value, np.uint8)
            storage[: height // 2, :, :3] = 240
            frame = storage[:, ::2]
            original = frame.copy()
            video.append(frame)
            np.testing.assert_array_equal(frame, original)
            frames.append(frame[..., :3].copy())
        encoded = video.encoded_size
        assert video.frames == 3
        assert video.size == size
        if pixel_format == "yuv444p":
            assert encoded == size
        else:
            assert encoded == (width + width % 2, height + height % 2)
    video.close()
    with pytest.raises(RuntimeError, match="after close"):
        video.append(frames[0])
    reader = imageio_ffmpeg.read_frames(str(path))
    with contextlib.closing(reader):
        metadata = next(reader)
        assert metadata["size"] == encoded
        assert metadata["fps"] == pytest.approx(12.5)
        decoded = [
            np.frombuffer(raw, np.uint8).reshape(encoded[1], encoded[0], 3) for raw in reader
        ]
    assert len(decoded) == len(frames)
    for actual, expected in zip(decoded, frames, strict=True):
        assert np.max(np.abs(actual[:height, :width].astype(int) - expected.astype(int))) < 8
        if padded:
            assert np.max(np.abs(actual[-1].astype(int) - actual[-2].astype(int))) < 8
    # Check the encoded stream, not merely the requested arguments.
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert pixel_format in probe.stderr


@pytest.mark.integration
def test_video_closing_an_empty_recording_reports_failure(tmp_path):
    video = VideoRecorder(tmp_path / "empty.mp4", (32, 24))
    with pytest.raises(RuntimeError, match="No frames were written"):
        video.close()
    video.close()


@pytest.mark.integration
def test_encoder_exit_failure_includes_stderr_even_for_one_small_frame(tmp_path):
    # A tiny write can fit in the OS pipe before FFmpeg fails; close must check exit status.
    with (
        pytest.raises(RuntimeError, match="mojive_missing_encoder") as caught,
        VideoRecorder(tmp_path / "bad.mp4", (2, 2), codec="mojive_missing_encoder") as video,
    ):
        video.append(np.zeros((2, 2, 3), np.uint8))
    assert "FFmpeg exit" in str(caught.value)
    assert "Unknown encoder" in str(caught.value)
    assert video._process is None
    assert video._stderr.closed


@pytest.mark.integration
def test_broken_encoder_pipe_stops_and_reaps_the_process(tmp_path):
    video = VideoRecorder(tmp_path / "bad.mp4", (1024, 1024), codec="mojive_missing_encoder")
    process = video._process
    with pytest.raises(RuntimeError, match="Unknown encoder"):
        for _ in range(16):
            video.append(np.zeros((1024, 1024, 3), np.uint8))
    assert process.poll() is not None
    assert video._process is None
    assert video._stderr.closed


@pytest.mark.integration
def test_video_cleanup_does_not_replace_a_caller_exception(tmp_path):
    with (
        pytest.raises(ValueError, match="invalid rollout") as caught,
        VideoRecorder(tmp_path / "empty.mp4", (32, 24)),
    ):
        raise ValueError("invalid rollout")
    assert "No frames were written" in caught.value.__notes__[0]


@pytest.mark.integration
def test_video_rejects_bad_frame_shape_then_can_continue(tmp_path):
    with VideoRecorder(tmp_path / "valid.mp4", (32, 24)) as video:
        for shape in ((24, 32), (24, 32, 2), (32, 24, 3)):
            with pytest.raises(ValueError, match="video frames must be"):
                video.append(np.zeros(shape, np.uint8))
        video.append(np.zeros((24, 32, 3), np.uint8))
