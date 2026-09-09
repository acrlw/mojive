# Use OpenGL's bottom-left offscreen coordinates, including the rasterizer's
# boundary ownership. Reflecting sample locations alone cannot match edge ties.
# Swap chains retain Vulkan's top-left presentation; public rectangles stay top-left.
function(mojive_vk_replace original replacement)
    string(FIND "${vulkan_code}" "${original}" position)
    if(position LESS 0)
        message(FATAL_ERROR "Reconcile the Vulkan build-tree patch with updated bgfx source")
    endif()
    string(REPLACE "${original}" "${replacement}" updated "${vulkan_code}")
    set(vulkan_code "${updated}" PARENT_SCOPE)
endfunction()
mojive_vk_replace("g_caps.vendorId = uint16_t(m_deviceProperties.vendorID);"
    "g_caps.originBottomLeft = true;\n                g_caps.vendorId = uint16_t(m_deviceProperties.vendorID);")
mojive_vk_replace("murmur.add(frameBuffer.m_renderPassHashKey);"
    "murmur.add(frameBuffer.m_renderPassHashKey);\n            murmur.add(frameBuffer.isSwapChain());")
mojive_vk_replace("setRasterizerState(rasterizationState, _state, m_wireframe);" [=[
setRasterizerState(rasterizationState, _state, m_wireframe);
            if (!frameBuffer.isSwapChain())
                rasterizationState.frontFace = rasterizationState.frontFace == VK_FRONT_FACE_CLOCKWISE
                    ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;]=])
mojive_vk_replace("vp.y        =  float(rect.m_y + rect.m_height);"
    "vp.y        = fb.isSwapChain() ? float(rect.m_y + rect.m_height) : float(fb.m_height - rect.m_y - rect.m_height);")
mojive_vk_replace("vp.height   = -float(rect.m_height);"
    "vp.height   = fb.isSwapChain() ? -float(rect.m_height) : float(rect.m_height);")
mojive_vk_replace("vp.y        =  float(height);"
    "vp.y        = frameBuffer.isSwapChain() ? float(height) : 0.f;")
mojive_vk_replace("vp.height   = -float(height);"
    "vp.height   = frameBuffer.isSwapChain() ? -float(height) : float(height);")
mojive_vk_replace("vkCmdSetScissor(m_commandBuffer, 0, 1, &rc);" "setFramebufferScissor(rc);")
mojive_vk_replace("vkCmdBeginRenderPass(m_commandBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);"
    "beginFramebufferPass(rpbi);")
mojive_vk_replace("vkCmdClearAttachments(m_commandBuffer, mrt, attachments, BX_COUNTOF(rect), rect);"
    "rect[0].rect = framebufferRect(rect[0].rect);\n                vkCmdClearAttachments(m_commandBuffer, mrt, attachments, BX_COUNTOF(rect), rect);")
mojive_vk_replace("\t\tvoid dbgTextRenderBegin(TextVideoMemBlitter& _blitter, FrameBufferHandle _handle) override" [=[
        VkRect2D framebufferRect(VkRect2D rect)
        {
            const auto& fb = getFrameBuffer(m_fbh);
            if (!fb.isSwapChain())
                rect.offset.y = int32_t(fb.m_height) - rect.offset.y - int32_t(rect.extent.height);
            return rect;
        }
        void setFramebufferScissor(VkRect2D rect)
        {
            rect = framebufferRect(rect);
            vkCmdSetScissor(m_commandBuffer, 0, 1, &rect);
        }
        void beginFramebufferPass(VkRenderPassBeginInfo info)
        {
            info.renderArea = framebufferRect(info.renderArea);
            vkCmdBeginRenderPass(m_commandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
        }
		void dbgTextRenderBegin(TextVideoMemBlitter& _blitter, FrameBufferHandle _handle) override]=])
