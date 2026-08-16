from forge_viewer.render.forge.text import TextRenderer


def test_text_atlas_keeps_pil_top_at_the_glyph_top():
    renderer = TextRenderer()
    renderer._chars = {"F"}
    renderer._build_atlas()

    assert renderer._glyphs["F"].uv[1] < renderer._glyphs["F"].uv[3]
