from __future__ import annotations

import ctypes
import ctypes.util
import sys
import weakref
from typing import Any

_views: dict[int, weakref.ReferenceType[Any]] = {}
_classes: set[int] = set()
_callbacks: tuple[Any, ...] = ()


def install(glfw: Any, handle: Any, owner: Any) -> int:
    if sys.platform != "darwin" or not hasattr(glfw, "get_cocoa_window"):
        return 0
    runtime = _runtime()
    window = glfw.get_cocoa_window(handle)
    view = runtime.objc_msgSend(window, runtime.sel_registerName(b"contentView"))
    if not view:
        return 0
    _views[int(view)] = weakref.ref(owner)
    _install_callbacks(runtime, runtime.object_getClass(view))
    return int(view)


def uninstall(token: int) -> None:
    _views.pop(int(token), None)


def _runtime():
    library = ctypes.util.find_library("objc")
    if not library:
        raise RuntimeError("Objective-C runtime is unavailable")
    runtime = ctypes.CDLL(library)
    runtime.sel_registerName.argtypes = [ctypes.c_char_p]
    runtime.sel_registerName.restype = ctypes.c_void_p
    runtime.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.objc_msgSend.restype = ctypes.c_void_p
    runtime.object_getClass.argtypes = [ctypes.c_void_p]
    runtime.object_getClass.restype = ctypes.c_void_p
    runtime.class_getInstanceMethod.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.class_getInstanceMethod.restype = ctypes.c_void_p
    runtime.method_getTypeEncoding.argtypes = [ctypes.c_void_p]
    runtime.method_getTypeEncoding.restype = ctypes.c_char_p
    runtime.class_addMethod.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    runtime.class_addMethod.restype = ctypes.c_bool
    runtime.method_setImplementation.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.method_setImplementation.restype = ctypes.c_void_p
    return runtime


def _install_callbacks(runtime, cls) -> None:
    global _callbacks
    class_id = int(cls)
    if class_id in _classes:
        return

    enter_type = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    exit_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)

    @enter_type
    def entered(view, _selector, _sender):
        _set_active(view, True)
        return 1

    @exit_type
    def exited(view, _selector, _sender):
        _set_active(view, False)

    _replace(runtime, cls, b"draggingEntered:", entered)
    _replace(runtime, cls, b"draggingExited:", exited)
    _callbacks = (*_callbacks, entered, exited)
    _classes.add(class_id)


def _replace(runtime, cls, name: bytes, callback) -> None:
    selector = runtime.sel_registerName(name)
    method = runtime.class_getInstanceMethod(cls, selector)
    if not method:
        return
    implementation = ctypes.cast(callback, ctypes.c_void_p)
    encoding = runtime.method_getTypeEncoding(method)
    if not runtime.class_addMethod(cls, selector, implementation, encoding):
        runtime.method_setImplementation(method, implementation)


def _set_active(view, active: bool) -> None:
    reference = _views.get(int(view))
    owner = reference() if reference is not None else None
    if owner is not None:
        owner._file_drag_active = bool(active)
