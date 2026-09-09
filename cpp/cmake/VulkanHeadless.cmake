# Peer windows must receive reset flags even when there is no main swap chain.
# Keep the pinned dependency untouched and check the build-tree patch context.
set(vulkan_source "${BGFX_DIR}/src/renderer_vk.cpp")
file(READ "${vulkan_source}" vulkan_code)
set(original [=[			if (!m_backBuffer.isSwapChain() )
			{
				if (m_mainSwapChain.width  != _swapChain.width
				||  m_mainSwapChain.height != _swapChain.height)]=])
set(replacement [=[			if (!m_backBuffer.isSwapChain() )
			{
				const bool vsyncChanged = 0 != ((_reset ^ m_reset) & BGFX_RESET_VSYNC);
				m_reset = _reset;
				if (vsyncChanged)
				{
					for (uint16_t ii = 0; ii < m_numWindows; ++ii)
					{
						if (isValid(m_windows[ii]) )
						{
							FrameBufferVK& fb = m_frameBuffers[m_windows[ii].idx];
							const SwapChain desc = fb.m_swapChain.m_desc;
							fb.update(m_commandBuffer, desc);
						}
					}
				}
				if (m_mainSwapChain.width  != _swapChain.width
				||  m_mainSwapChain.height != _swapChain.height)]=])
string(FIND "${vulkan_code}" "${original}" match)
if(match LESS 0)
    message(FATAL_ERROR "Reconcile the Vulkan peer VSync fix with the updated bgfx source")
endif()
string(REPLACE "${original}" "${replacement}" vulkan_code "${vulkan_code}")
include("${CMAKE_CURRENT_LIST_DIR}/VulkanRasterization.cmake")
set(patched_vulkan "${CMAKE_BINARY_DIR}/patched/renderer_vk.cpp")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/patched")
file(CONFIGURE OUTPUT "${patched_vulkan}" CONTENT "${vulkan_code}" @ONLY)
get_target_property(bgfx_sources bgfx SOURCES)
list(REMOVE_ITEM bgfx_sources "${vulkan_source}")
list(APPEND bgfx_sources "${patched_vulkan}")
set_property(TARGET bgfx PROPERTY SOURCES "${bgfx_sources}")
target_include_directories(bgfx PRIVATE "${BGFX_DIR}/src")
