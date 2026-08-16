import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"


CPU_TESTS = "tests/test_scene.py"
CAM_TESTS = "tests/test_camera.py"
VC_TESTS = "tests/test_viewcube.py"
PERTURB_TESTS = "tests/test_perturb.py"
REFL_TESTS = "tests/test_reflection.py"
REFL_GPU = "tests/gpu/test_reflection.py"
GPU_TESTS = "tests/gpu/test_forge_core.py"
UI_TESTS = "tests/gpu/test_ui_interaction.py"


CASES = [
    (
        "clear_main 不先打开深度写",
        "src/forge_viewer/render/forge/targets.py",
        "        self.fbo.depth_mask = True\n        self.fbo.use()",
        "        self.fbo.use()",
        "test_depth_mask_replay_is_defused",
        GPU_TESTS,
    ),
    (
        "实例基址恒为 0（等于没有字节偏移）",
        "src/forge_viewer/render/forge/instances.py",
        "vao.glo, self.buffer.glo, INSTANCE_BYTES, start * INSTANCE_BYTES, attrs",
        "vao.glo, self.buffer.glo, INSTANCE_BYTES, 0, attrs",
        "test_both_instance_strategies_draw_the_same_thing",
        GPU_TESTS,
    ),
    (
        "整数附件改用库的 clear()",
        "src/forge_viewer/render/forge/targets.py",
        "        if depth_done and self._gl.clear_color_uint(self.id_draw_buffer, int(value)):\n            return",
        "        self.id_fbo.clear(float(value), 0.0, 0.0, 0.0)\n        return",
        "test_integer_attachment_clear_is_exact",
        GPU_TESTS,
    ),
    (
        "解绑改成裸 ctx.screen.use()",
        "src/forge_viewer/render/forge/state_guard.py",
        "    viewport = ctx.viewport\n    scissor = ctx.scissor\n    ctx.screen.use()\n    ctx.viewport = viewport\n    ctx.scissor = scissor",
        "    ctx.screen.use()",
        "test_bind_default_framebuffer_does_not_touch_viewport",
        GPU_TESTS,
    ),
    (
        "状态守卫漏还 polygon_mode",
        "src/forge_viewer/render/forge/state_guard.py",
        '        gl.polygon_mode(s["polygon_mode"])',
        "        pass",
        "test_state_guard_restores_every_item",
        GPU_TESTS,
    ),
    (
        "容量按需精确分配（不翻倍）",
        "src/forge_viewer/render/forge/instances.py",
        "        new_cap = max(count, self.capacity * 2, 64)",
        "        new_cap = count",
        "test_capacity_grows_by_doubling",
        GPU_TESTS,
    ),
    (
        "打包不转置（行主序直接上传）",
        "src/forge_viewer/render/forge/instances.py",
        "        dst[:, 0:16] = scene.transforms.transpose(0, 2, 1).reshape(n, 16)",
        "        dst[:, 0:16] = scene.transforms.reshape(n, 16)",
        "test_pack_transposes_row_major_to_column_major",
        GPU_TESTS,
    ),
    (
        "着色器编译失败时不保留上一份",
        "src/forge_viewer/render/forge/programs.py",
        "                continue\n            old = self._programs.get(key)",
        "                self._programs[key] = None\n                continue\n            old = self._programs.get(key)",
        "test_shader_compile_failure_keeps_last_good_program",
        GPU_TESTS,
    ),
    (
        "uniform 缓存不比较（每次都写）",
        "src/forge_viewer/render/forge/programs.py",
        "        if self._cache.get(name, _MISSING) == value:\n            return",
        "        pass",
        "test_uniform_cache_skips_unchanged_writes",
        GPU_TESTS,
    ),
    (
        "GPU 计时不可用时也报表（假数字）",
        "src/forge_viewer/render/forge/timing.py",
        "        if not self.gpu_available:\n            return {}",
        "        pass",
        "test_gpu_timing_degrades_to_empty_table",
        GPU_TESTS,
    ),
    (
        "object_id 的属性类型声明成 float",
        "src/forge_viewer/render/forge/instances.py",
        '("in_object_id", "1u", 4, 1, 112, G.GL_UNSIGNED_INT),',
        '("in_object_id", "1f", 4, 1, 112, G.GL_FLOAT),',
        "test_object_id_reaches_the_shader_exactly",
        GPU_TESTS,
    ),
    (
        "object_id 经 float32 中转（重新引入 2²⁴ 上限）",
        "src/forge_viewer/render/forge/instances.py",
        "        self._raw[:n, 28] = scene.object_id",
        "        self._raw[:n, 28] = scene.object_id.astype(np.float32).astype(np.uint32)"
        + "\n        self._raw[:n, 28] = np.float32(scene.object_id).astype(np.uint32)",
        "test_object_id_survives_packing_as_an_exact_uint32",
        GPU_TESTS,
    ),
    (
        "桶身份不含透明标志（同键的透明与不透明挤一个桶）",
        "src/forge_viewer/render/scene.py",
        '            key = (*row["key"], float(row["color"][3]) < 1.0)',
        '            key = (*row["key"], False)',
        "test_same_mesh_and_material_but_different_alpha_split_into_two_buckets",
        CPU_TESTS,
    ),
    (
        "写入索引退化成恒等映射",
        "src/forge_viewer/render/scene.py",
        "        write_index[order] = np.arange(n, dtype=np.int32)",
        "        write_index[:] = np.arange(n, dtype=np.int32)",
        "test_write_index_actually_places_each_instance_in_its_bucket",
        CPU_TESTS,
    ),
    (
        "透明桶按由近到远画",
        "src/forge_viewer/render/scene.py",
        "        keyed.sort(key=lambda t: -t[0])",
        "        keyed.sort(key=lambda t: t[0])",
        "test_transparent_buckets_draw_far_to_near",
        CPU_TESTS,
    ),
    (
        "不透明桶没有排在透明桶前面",
        "src/forge_viewer/render/scene.py",
        "        order_of_bucket = sorted(range(len(ident)), key=lambda b: (ident[b][2], b))",
        "        order_of_bucket = sorted(range(len(ident)), key=lambda b: (not ident[b][2], b))",
        "test_opaque_buckets_all_come_before_transparent_ones",
        CPU_TESTS,
    ),
    (
        "倒影不翻绕向（看到的是背面）",
        "src/forge_viewer/render/forge/passes/reflect.py",
        '        gl.front_face = "cw"',
        "        pass",
        "test_reflection_shows_outer_faces_not_the_inside",
        REFL_GPU,
    ),
    (
        "不开斜裁剪面（地板底下的东西翻进倒影）",
        "src/forge_viewer/render/forge/passes/reflect.py",
        "        gl.enable_direct(GL_CLIP_DISTANCE0)",
        "        pass",
        "test_geometry_below_the_plane_stays_out_of_the_reflection",
        REFL_GPU,
    ),
    (
        "反射图清成背景色而不是黑（整块地板被提亮）",
        "src/forge_viewer/render/forge/passes/reflect.py",
        "        self.fbo.clear(0.0, 0.0, 0.0, 1.0)",
        "        self.fbo.clear(*ctx.background)",
        "test_reflection_is_added_not_mixed",
        REFL_GPU,
    ),
    (
        "倒影写成 mix 而不是加法",
        "src/forge_viewer/render/forge/shaders/scene_body.glsl",
        "        lit += v.reflect * texture(u_reflection, gl_FragCoord.xy / u_reflection_size).rgb;",
        "        lit = mix(lit, texture(u_reflection, gl_FragCoord.xy / u_reflection_size).rgb, v.reflect);",
        "test_reflection_is_added_not_mixed",
        REFL_GPU,
    ),
    (
        "反射系数不乘进去（写死成常数）",
        "src/forge_viewer/render/forge/shaders/scene_body.glsl",
        "        lit += v.reflect * texture(u_reflection",
        "        lit += 0.35 * texture(u_reflection",
        "test_reflection_scales_with_reflectance",
        REFL_GPU,
    ),
    (
        "反射面的法线直接取模型矩阵第三列（不走逆转置）",
        "src/forge_viewer/render/forge/passes/reflect.py",
        "            normal = np.linalg.inv(basis).T @ np.array([0.0, 0.0, 1.0])",
        "            normal = basis @ np.array([0.0, 0.0, 1.0])",
        "test_normal_uses_the_inverse_transpose_not_the_third_column",
        REFL_TESTS,
    ),
    (
        "反射面取第一个而不是最强的那个",
        "src/forge_viewer/render/forge/passes/reflect.py",
        "        idx = int(np.argmax(refl))",
        "        idx = int(np.nonzero(refl >= 0.0)[0][0])",
        "test_the_strongest_reflector_wins",
        REFL_TESTS,
    ),
    (
        "镜像矩阵的平移项少一半",
        "src/forge_viewer/math3d.py",
        "    out[:3, 3] = 2.0 * d * n",
        "    out[:3, 3] = d * n",
        "test_mirror_keeps_points_on_the_plane_put",
        REFL_TESTS,
    ),
    (
        "不建出厂停靠布局",
        "src/forge_viewer/ui/window.py",
        "            self._build_default_layout()",
        "            pass",
        "test_viewport_gets_real_estate",
        UI_TESTS,
    ),
    (
        "字号跟 DPI 走（点空间里放大两次）",
        "src/forge_viewer/ui/window.py",
        "self.font_report = fonts.load(imgui, io, size_pt=self.config.font_size_pt)",
        "self.font_report = fonts.load(\n            imgui, io, size_pt=self.config.font_size_pt * self._ui_scale\n        )",
        "test_font_size_is_in_layout_space",
        UI_TESTS,
    ),
    (
        "轴球尺寸跟 DPI 走",
        "src/forge_viewer/ui/app.py",
        "self.camera.view(), rect, cursor, self.window.style_scale",
        "self.camera.view(), rect, cursor, self.window.ui_scale",
        "test_view_gizmo_fits_the_corner",
        UI_TESTS,
    ),
    (
        "轴球投影拿错基向量（用 forward 当 right）",
        "src/forge_viewer/ui/viewcube.py",
        "sx = center[0] + float(np.dot(world, right)) * radius_pt",
        "sx = center[0] + float(np.dot(world, forward)) * radius_pt",
        "test_view_gizmo_axis_points_at_you_when_you_look_down_it",
        UI_TESTS,
    ),
    (
        "点球去的机位轴序接反",
        "src/forge_viewer/ui/viewcube.py",
        "    yaw = 0.0 if axis == 0 else 90.0",
        "    yaw = 90.0 if axis == 0 else 0.0",
        "test_view_gizmo_click_snaps_to_that_axis",
        UI_TESTS,
    ),
    (
        "光标→像素不翻 Y",
        "src/forge_viewer/types.py",
        "return max(col, 0), self.height - 1 - max(row_from_top, 0)",
        "return max(col, 0), max(row_from_top, 0)",
        "test_click_picks_the_object_actually_under_the_cursor",
        UI_TESTS,
    ),
    (
        "不 merge CJK 字体（中文变方框）",
        "src/forge_viewer/ui/fonts.py",
        "        io.fonts.add_font_from_file_ttf(cjk[0], size_pt, cfg(merge=True, index=cjk[1]))\n        rep.cjk = label",
        "        rep.cjk = label",
        "test_cjk_glyphs_present",
        UI_TESTS,
    ),
    (
        "界面字体换成比例字体（字面不等宽）",
        "src/forge_viewer/ui/fonts.py",
        """    mono, label = _resolve(
        _MONO,
        _REMOTE_MONO,""",
        """    from imgui_bundle import imgui_bundle_folder

    mono = (str(Path(imgui_bundle_folder()) / "assets/fonts/Roboto/Roboto-Regular.ttf"), 0)
    label = "Roboto"
    _unused = _resolve(
        _MONO,
        _REMOTE_MONO,""",
        "test_font_is_monospace",
        UI_TESTS,
    ),
    (
        "字改回 add_text（落回 imgui 的整点量化）",
        "src/forge_viewer/ui/viewcube.py",
        """    pen_x = pos.x - (box[0] + box[2]) * 0.5
    pen_y = pos.y - (box[1] + box[3]) * 0.5""",
        """    dl.add_text(font, size, imgui.ImVec2(pos.x - size * 0.32, pos.y - size * 0.5), color, text)
    pen_x = pos.x - (box[0] + box[2]) * 0.5
    pen_y = pos.y - (box[1] + box[3]) * 0.5
    return  # noqa""",
        "test_gizmo_label_sits_in_the_middle_of_its_ball",
        UI_TESTS,
    ),
    (
        "单元不按深度画（改按轴序），线/球/字交错",
        "src/forge_viewer/ui/viewcube.py",
        "        for i, b in enumerate(self._balls):  # `layout()` 已按从远到近排好",
        "        for i, b in enumerate(sorted(self._balls, key=lambda x: x.axis)):  #",
        "test_gizmo_draws_each_axis_as_one_unit",
        UI_TESTS,
    ),
    (
        "负向球退回半透明",
        "src/forge_viewer/ui/viewcube.py",
        "imgui.ImVec4(*face, 1.0)",
        "imgui.ImVec4(*face, 0.55)",
        "test_negative_balls_are_dark_and_opaque",
        UI_TESTS,
    ),
    (
        "悬停把球画大",
        "src/forge_viewer/ui/viewcube.py",
        "            dl.add_circle_filled(pos, b.radius, u32(imgui.ImVec4(*face, 1.0)), 24)",
        "            dl.add_circle_filled(\n                pos, b.radius * (1.25 if hovered else 1.0), "
        "u32(imgui.ImVec4(*face, 1.0)), 24\n            )",
        "test_hover_does_not_resize_the_ball",
        UI_TESTS,
    ),
    (
        "俯视保留当前偏航（每次朝向都不一样）",
        "src/forge_viewer/ui/viewcube.py",
        "        return TOP_YAW, PITCH_LIMIT * (1.0 if sign > 0 else -1.0)",
        "        return current_yaw, PITCH_LIMIT * (1.0 if sign > 0 else -1.0)",
        "test_top_view_is_canonical_x_right_y_up",
        UI_TESTS,
    ),
    (
        "抖一个像素就当拖拽（缓动被掐死在半路）",
        "src/forge_viewer/ui/app.py",
        "        settled = self.router.travel >= CLICK_SLOP_PT",
        "        settled = True",
        "test_clicking_during_a_transition_does_not_strand_the_camera",
        UI_TESTS,
    ),
    (
        "关窗口时无条件 glfw.terminate（连累另一个窗口）",
        "src/forge_viewer/ui/window.py",
        "            if _live_windows == 0:\n                # **只有最后一个窗口才 terminate**，理由见 `_live_windows`\n                glfw.terminate()",
        "            glfw.terminate()",
        "test_closing_one_window_leaves_glfw_alive_for_the_other",
        UI_TESTS,
    ),
    (
        "关窗口前不切回自己的 GL 上下文",
        "src/forge_viewer/ui/window.py",
        "            glfw.make_context_current(self._window)\n        except Exception as e:",
        "            pass\n        except Exception as e:",
        "test_closing_one_window_leaves_glfw_alive_for_the_other",
        UI_TESTS,
    ),
    (
        "让路只有一档（一转视角手柄就消失）",
        "src/forge_viewer/ui/app.py",
        "        yielding = gs.gizmo_yields(self._state)\n        interactive = self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO)",
        "        interactive = self.router.claim in (gs.Claim.NONE, gs.Claim.OBJECT_GIZMO)\n        yielding = gs.gizmo_yields(self._state) or not interactive",
        "test_gizmo_stays_drawn_while_the_camera_is_being_dragged",
        UI_TESTS,
    ),
    (
        "轴色换成纯三原色（丢掉亮度配平）",
        "src/forge_viewer/ui/theme.py",
        '    "x": rgb8(239, 110, 106),',
        '    "x": (0.666, 0.0, 0.0, 1.0),',
        "test_axis_colors_are_luminance_balanced",
        "tests/test_theme.py",
    ),
    (
        "键盘闸判 want_capture_keyboard（开了键盘导航就恒为真，快捷键全死）",
        "src/forge_viewer/ui/app.py",
        "        if io.want_text_input:",
        "        if io.want_capture_keyboard:",
        "test_the_keyboard_shortcuts_are_not_swallowed",
        UI_TESTS,
    ),
    (
        "不给 ImGuizmo 指定悬停窗口（手柄画得出来但点不着）",
        "src/forge_viewer/ui/gizmo.py",
        "        if self._window is not None:\n            gz.set_alternative_window(self._window)",
        "        if False:\n            gz.set_alternative_window(self._window)",
        "test_dragging_the_gizmo_moves_the_object_not_the_camera",
        UI_TESTS,
    ),
    (
        "手柄根本没调 manipulate（判定说该出现，实际什么都没有）",
        "src/forge_viewer/ui/gizmo.py",
        "        changed = gz.manipulate(view, proj, op, gz.MODE.local, m)",
        "        changed = False",
        "test_gizmo_is_live_for_a_free_body",
        UI_TESTS,
    ),
    (
        "pivot 退回普通字段（写了不置脏、不下发）",
        "src/forge_viewer/ui/camera.py",
        "        self._pivot = np.asarray(value, np.float64).reshape(3).copy()\n        self._touch()",
        "        self._pivot = np.asarray(value, np.float64).reshape(3).copy()",
        "test_every_public_setter_marks_dirty_and_publishes",
        CAM_TESTS,
    ),
    (
        "虚线的刀口不做透视校正（屏幕上又长又短）",
        "src/forge_viewer/ui/perturb.py",
        "        t0 = u0 * wa / ((1.0 - u0) * wb + u0 * wa)\n        t1 = u1 * wa / ((1.0 - u1) * wb + u1 * wa)",
        "        t0 = u0\n        t1 = u1",
        "test_dashes_are_uniform_in_screen_space",
        PERTURB_TESTS,
    ),
    (
        "剪影不走屏幕等距那条路（算得对但没用上）",
        "src/forge_viewer/ui/perturb.py",
        "            segs.extend(dash_screen(a, b, cam, rect))",
        "            segs.extend(dash(a, b))",
        "test_published_silhouette_uses_the_screen_space_dash",
        PERTURB_TESTS,
    ),
    (
        "缓动换回 smoothstep（两头都缓，起步肉）",
        "src/forge_viewer/ui/camera.py",
        "    return t * (2.0 - t)",
        "    return t * t * (3.0 - 2.0 * t)",
        "test_easing_is_ease_out_not_ease_in_out",
        CAM_TESTS,
    ),
    (
        "球心被盖住时照样画字（-Y 球上冒出一个 Y）",
        "src/forge_viewer/ui/viewcube.py",
        "        reach = other.radius + me.radius * LABEL_HIDE_SLACK",
        "        reach = 0.0",
        "test_label_is_dropped_when_the_ball_center_is_covered",
        VC_TESTS,
    ),
    (
        "轴球做近大远小（伪透视）",
        "src/forge_viewer/ui/viewcube.py",
        "                    radius=ball_pt,  #",
        "                    radius=ball_pt * (1.0 - 0.18 * depth),  #",
        "test_all_balls_are_the_same_size",
        VC_TESTS,
    ),
    (
        "轴球布局不按深度排（远的盖住近的）",
        "src/forge_viewer/ui/viewcube.py",
        "    out.sort(key=lambda b: -b.depth)",
        "    out.sort(key=lambda b: b.depth)",
        "test_layout_is_sorted_far_to_near",
        VC_TESTS,
    ),
    (
        "命中取远的那个球",
        "src/forge_viewer/ui/viewcube.py",
        "and (best is None or b.depth < best.depth)",
        "and (best is None or b.depth > best.depth)",
        "test_hit_test_prefers_the_nearer_ball",
        VC_TESTS,
    ),
]


def run(test: str, path: str) -> bool:

    marks = "gpu" if path.startswith("tests/gpu") else "not gpu"
    env = dict(os.environ)

    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run(
        [
            str(PY),
            "-m",
            "pytest",
            path,
            "-m",
            marks,
            "-k",
            test,
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-header",
            "-x",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    return r.returncode == 0


def main() -> int:
    bad = []
    for desc, relpath, old, new, test, tpath in CASES:
        path = ROOT / relpath
        backup = path.read_text()
        if old not in backup:
            print(f"?? {desc:38} 找不到要替换的原文——反向验证本身失效了")
            bad.append(desc)
            continue
        stat = path.stat()
        try:
            path.write_text(backup.replace(old, new, 1))
            still_green = run(test, tpath)
        finally:
            path.write_text(backup)

            os.utime(path, (stat.st_atime, stat.st_mtime))
        mark = "✗" if still_green else "✓"
        print(f"{mark} {desc:38} → {'仍然全绿（判据没守住！）' if still_green else '变红'}")
        if still_green:
            bad.append(desc)

    shutil.rmtree(ROOT / ".pytest_cache", ignore_errors=True)
    print()
    for path, marks in ((CPU_TESTS, "not gpu"), (GPU_TESTS, "gpu")):
        r = subprocess.run(
            [
                str(PY),
                "-m",
                "pytest",
                path,
                "-m",
                marks,
                "-q",
                "-p",
                "no:cacheprovider",
                "--no-header",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(f"恢复之后 {path:32} {r.stdout.strip().splitlines()[-1]}")
    if bad:
        print(f"\n{len(bad)} 条判据没守住：{bad}")
        return 1
    print(f"\n{len(CASES)} 条判据全部通过反向验证。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
