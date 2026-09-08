# The pinned bgfx Metal backend ties sampler anisotropy to a window swap chain.
# Offscreen renderers have none. Compile a narrowly patched build-tree copy;
# keep the locked upstream submodule untouched and fail if its context changes.
set(metal_source "${BGFX_DIR}/src/renderer_mtl.cpp")
file(READ "${metal_source}" metal_code)
set(original [=[desc->setMaxAnisotropy(true
					&& NULL != m_mainFrameBuffer.m_swapChain
					&& (0 != (_flags & (BGFX_SAMPLER_MIN_ANISOTROPIC|BGFX_SAMPLER_MAG_ANISOTROPIC) ) )
						? m_mainFrameBuffer.m_swapChain->m_maxAnisotropy
						: 1
					);]=])
string(FIND "${metal_code}" "${original}" match)
if(match LESS 0)
    message(FATAL_ERROR "Reconcile the Metal headless anisotropy fix with the updated bgfx source")
endif()
string(REPLACE "${original}" [=[desc->setMaxAnisotropy(
                    (_flags & (BGFX_SAMPLER_MIN_ANISOTROPIC|BGFX_SAMPLER_MAG_ANISOTROPIC))
                        ? m_samplerAnisotropy : 1);]=] metal_code "${metal_code}")
string(REPLACE "m_depthClamp = m_supportsDepthClipMode" "m_samplerAnisotropy = (_reset & BGFX_RESET_MAXANISOTROPY) ? 16 : 1;
            m_depthClamp = m_supportsDepthClipMode" metal_code "${metal_code}")
string(REPLACE "uint32_t  m_reset;" "uint32_t  m_reset;
        uint32_t m_samplerAnisotropy = 1;" metal_code "${metal_code}")
set(patched_metal "${CMAKE_BINARY_DIR}/patched/renderer_mtl.cpp")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/patched")
file(CONFIGURE OUTPUT "${patched_metal}" CONTENT "${metal_code}" @ONLY)
get_target_property(bgfx_sources bgfx SOURCES)
list(REMOVE_ITEM bgfx_sources "${metal_source}")
list(APPEND bgfx_sources "${patched_metal}")
set_property(TARGET bgfx PROPERTY SOURCES "${bgfx_sources}")
target_include_directories(bgfx PRIVATE "${BGFX_DIR}/src")
