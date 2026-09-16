"""Isolated GTK3 X11 selection owner; exits when the clipboard is replaced."""

from __future__ import annotations

import ctypes as c
import ctypes.util
import sys
from pathlib import Path

from .clipboard import file_targets, image_png


class _Target(c.Structure):
    _fields_ = [("target", c.c_char_p), ("flags", c.c_uint), ("info", c.c_uint)]


def serve(path: Path, kind: str) -> None:
    library = ctypes.util.find_library("gtk-3")
    if library is None:
        raise RuntimeError("X11 clipboard sharing requires GTK3")
    gtk = c.CDLL(library)

    def function(name, result, *arguments):
        value = getattr(gtk, name)
        value.restype = result
        value.argtypes = arguments
        return value

    pointer, integer = c.c_void_p, c.c_int
    init = function("gtk_init_check", integer, pointer, pointer)
    if not init(None, None):
        raise RuntimeError("Cannot connect to the X11 desktop clipboard")
    atom = function("gdk_atom_intern_static_string", pointer, c.c_char_p)(b"CLIPBOARD")
    board = function("gtk_clipboard_get", pointer, pointer)(atom)
    target_of = function("gtk_selection_data_get_target", pointer, pointer)
    write = function("gtk_selection_data_set", None, pointer, pointer, integer, c.c_char_p, integer)
    quit_loop = function("gtk_main_quit", None)
    loop_level = function("gtk_main_level", c.c_uint)
    get_type = c.CFUNCTYPE(None, pointer, pointer, c.c_uint, pointer)
    clear_type = c.CFUNCTYPE(None, pointer, pointer)
    entries = {"image/png": image_png(path)} if kind == "image" else file_targets(path)
    payloads = list(entries.values())
    names = [name.encode() for name in entries]
    targets = (_Target * len(names))(*(_Target(name, 0, i) for i, name in enumerate(names)))
    cleared = False

    @get_type
    def provide(_board, selection, index, _data):
        payload = payloads[index]
        write(selection, target_of(selection), 8, payload, len(payload))

    @clear_type
    def clear(_board, _data):
        nonlocal cleared
        cleared = True
        if loop_level():
            quit_loop()

    own = function(
        "gtk_clipboard_set_with_data",
        integer,
        pointer,
        c.POINTER(_Target),
        c.c_uint,
        get_type,
        clear_type,
        pointer,
    )
    if not own(board, targets, len(targets), provide, clear, None):
        raise RuntimeError("Cannot take ownership of the desktop clipboard")
    display = function("gdk_display_get_default", pointer)()
    function("gdk_display_flush", None, pointer)(display)
    print("READY", flush=True)
    if not cleared:
        function("gtk_main", None)()


if __name__ == "__main__":
    try:
        serve(Path(sys.argv[2]), sys.argv[1])
    except Exception as exc:
        print(str(exc), flush=True)
        sys.exit(1)
