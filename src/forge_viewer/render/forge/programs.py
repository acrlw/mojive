"""Shader program creation, preprocessing, and uniform caching."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import moderngl

from ...log import get_logger

log = get_logger("programs")

SHADER_DIR = Path(__file__).parent / "shaders"
_INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)


def _expand(path: Path, seen: set[Path] | None = None) -> tuple[str, set[Path]]:
    seen = seen if seen is not None else set()
    if path in seen:
        return "", seen
    seen.add(path)
    src = path.read_text(encoding="utf-8")
    out: list[str] = []
    pos = 0
    for m in _INCLUDE.finditer(src):
        out.append(src[pos : m.start()])
        inc_text, seen = _expand(path.parent / m.group(1), seen)
        out.append(inc_text)
        pos = m.end()
    out.append(src[pos:])
    return "".join(out), seen


def _inject_defines(src: str, defines: dict[str, object]) -> str:
    if not defines:
        return src
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#version"):
            block = [f"#define {k} {v}" for k, v in sorted(defines.items())]
            return "\n".join([*lines[: i + 1], *block, *lines[i + 1 :]])
    block = "\n".join(f"#define {k} {v}" for k, v in sorted(defines.items()))
    return block + "\n" + src


@dataclass
class ProgramSpec:
    name: str
    vertex: str
    fragment: str
    geometry: str | None = None
    defines: dict[str, object] = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.name, tuple(sorted(self.defines.items())))


class ProgramCache:
    def __init__(self, ctx: moderngl.Context, shader_dir: Path | None = None) -> None:
        self.ctx = ctx
        self.dir = shader_dir or SHADER_DIR
        self.generation = 0

        self._programs: dict[tuple, moderngl.Program] = {}
        self._specs: dict[tuple, ProgramSpec] = {}
        self._deps: dict[tuple, dict[Path, float]] = {}
        self._failed: dict[tuple, str] = {}

    def get(self, spec: ProgramSpec) -> moderngl.Program:
        key = spec.key()
        prog = self._programs.get(key)
        if prog is None:
            prog = self._compile(spec)
            self._programs[key] = prog
            self._specs[key] = spec
        return prog

    def _sources(self, spec: ProgramSpec) -> tuple[dict[str, str], dict[Path, float]]:
        deps: dict[Path, float] = {}
        out: dict[str, str] = {}
        for stage, fname in (
            ("vertex_shader", spec.vertex),
            ("fragment_shader", spec.fragment),
            ("geometry_shader", spec.geometry),
        ):
            if not fname:
                continue
            path = self.dir / fname
            text, seen = _expand(path)
            for p in seen:
                deps[p] = p.stat().st_mtime
            out[stage] = _inject_defines(text, spec.defines)
        return out, deps

    def _compile(self, spec: ProgramSpec) -> moderngl.Program:
        sources, deps = self._sources(spec)
        prog = self.ctx.program(**sources)
        self._deps[spec.key()] = deps
        self._failed.pop(spec.key(), None)
        return prog

    def reload_changed(self) -> list[str]:
        changed: list[str] = []
        for key, spec in list(self._specs.items()):
            deps = self._deps.get(key, {})
            if not any(p.exists() and p.stat().st_mtime != t for p, t in deps.items()):
                continue
            try:
                prog = self._compile(spec)
            except Exception as e:
                msg = str(e)
                if self._failed.get(key) != msg:
                    self._failed[key] = msg
                    log.error(
                        "Shader {} failed to compile; keeping the previous version:\n{}",
                        spec.name,
                        msg,
                    )

                self._deps[key] = {
                    p: (p.stat().st_mtime if p.exists() else t) for p, t in deps.items()
                }
                continue
            old = self._programs.get(key)
            self._programs[key] = prog
            if old is not None:
                old.release()
            changed.append(spec.name)
            log.info("Shader {} reloaded", spec.name)
        if changed:
            self.generation += 1
        return changed

    @property
    def last_error(self) -> str:
        return next(iter(self._failed.values()), "")

    def release(self) -> None:
        for prog in self._programs.values():
            prog.release()
        self._programs.clear()


class UniformCache:
    __slots__ = ("_cache", "_generation", "_members", "_program")

    def __init__(self, program: moderngl.Program, generation: int = 0) -> None:
        self._program = program
        self._generation = generation
        self._cache: dict[str, object] = {}
        self._members = set(program)

    def rebind(self, program: moderngl.Program, generation: int) -> None:
        if program is self._program and generation == self._generation:
            return
        self._program = program
        self._generation = generation
        self._cache.clear()
        self._members = set(program)

    def set(self, name: str, value) -> None:
        if name not in self._members:
            return
        if self._cache.get(name, _MISSING) == value:
            return
        self._program[name].value = value
        self._cache[name] = value

    def force(self, name: str, value) -> None:
        if name in self._members:
            self._program[name].value = value
            self._cache[name] = value

    def reset(self) -> None:
        self._cache.clear()


class _Missing:
    def __eq__(self, other) -> bool:
        return False

    __hash__ = None  # type: ignore[assignment]


_MISSING = _Missing()
