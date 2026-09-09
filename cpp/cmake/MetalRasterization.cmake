# Offscreen Metal uses the same bottom-left rasterization and edge ownership as
# OpenGL and Vulkan. Public view/scissor rectangles remain top-left; swap chains
# retain Metal presentation coordinates. No shader- or sample-position bias.
function(mojive_metal_replace original replacement)
    string(FIND "${metal_code}" "${original}" position)
    if(position LESS 0)
        message(FATAL_ERROR "Reconcile Metal rasterization with updated bgfx source")
    endif()
    string(REPLACE "${original}" "${replacement}" updated "${metal_code}")
    set(metal_code "${updated}" PARENT_SCOPE)
endfunction()
mojive_metal_replace("g_caps.gpu[0].vendorId = g_caps.vendorId;"
    "g_caps.originBottomLeft = true;\n            g_caps.gpu[0].vendorId = g_caps.vendorId;")
mojive_metal_replace("rce->setViewport(viewport);" "rce->setViewport(framebufferViewport(viewport));")
mojive_metal_replace("rce->setViewport(vp);" "rce->setViewport(framebufferViewport(vp));")
mojive_metal_replace("rce->setScissorRect(rc);" "rce->setScissorRect(framebufferScissor(rc));")
mojive_metal_replace("rce->setScissorRect(sciRect);" "rce->setScissorRect(framebufferScissor(sciRect));")
mojive_metal_replace("rce->setFrontFacingWinding( (newFlags&BGFX_STATE_FRONT_CCW)"
    "rce->setFrontFacingWinding( ((newFlags&BGFX_STATE_FRONT_CCW) != 0) != offscreenFramebuffer()")
mojive_metal_replace("bool hasDepth(FrameBufferHandle _fbh)" [=[
        bool offscreenFramebuffer()
        {
            return isValid(m_fbh) && !m_frameBuffers[m_fbh.idx].m_swapChain;
        }
        MTL::Viewport framebufferViewport(MTL::Viewport viewport)
        {
            if (offscreenFramebuffer())
            {
                viewport.originY = m_frameBuffers[m_fbh.idx].m_height - viewport.originY;
                viewport.height = -viewport.height;
            }
            return viewport;
        }
        MTL::ScissorRect framebufferScissor(MTL::ScissorRect rect)
        {
            if (offscreenFramebuffer())
                rect.y = m_frameBuffers[m_fbh.idx].m_height - rect.y - rect.height;
            return rect;
        }
        bool hasDepth(FrameBufferHandle _fbh)]=])
