from mojive.render.text import TextLayout


def test_text_atlas_keeps_pil_top_at_the_glyph_top():
    layout = TextLayout()
    layout._chars = {"F"}
    layout._build_atlas()

    assert layout._glyphs["F"].uv[1] < layout._glyphs["F"].uv[3]
