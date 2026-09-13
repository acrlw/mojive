#version 330 core

#ifndef ID_ATTACHMENT
#define ID_ATTACHMENT 0
#endif

flat in uint v_id;
#ifdef ID_MASK_FLOAT
layout(location = ID_ATTACHMENT) out float o_mask;
#else
layout(location = ID_ATTACHMENT) out uint o_id;
#endif

#ifdef ID_ONLY_SELECTED
uniform uint u_selected;
#endif

void main() {
#ifdef ID_ONLY_SELECTED
    if (v_id != u_selected) discard;
#endif
#ifdef ID_MASK_FLOAT
    o_mask = 1.0;
#else
    o_id = v_id;
#endif
}
