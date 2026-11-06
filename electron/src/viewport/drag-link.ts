import { dragLinkFragment } from './drag-link-shader';
import type { P } from './geometry';
/** Render the production implicit G3 connector shader at framebuffer resolution. */
export class DragLinkRaster {
  private canvas = document.createElement('canvas');
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private buffer: WebGLBuffer;
  constructor() {
    const gl = this.canvas.getContext('webgl2', {
      alpha: true,
      premultipliedAlpha: true,
      antialias: false,
    });
    if (!gl) throw new Error('WebGL2 is required for drag feedback');
    this.gl = gl;
    const compile = (kind: number, source: string) => {
      const shader = gl.createShader(kind)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader) || 'Drag link shader compilation failed');
      return shader;
    };
    const vertex = compile(
        gl.VERTEX_SHADER,
        '#version 300 es\nin vec2 position;uniform vec2 extent;out vec2 v_pixel;void main(){v_pixel=vec2((position.x+1.)*.5,(1.-position.y)*.5)*extent;gl_Position=vec4(position,0.,1.);}',
      ),
      fragment = compile(gl.FRAGMENT_SHADER, dragLinkFragment);
    this.program = gl.createProgram()!;
    gl.attachShader(this.program, vertex);
    gl.attachShader(this.program, fragment);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(this.program) || 'Drag link link failed');
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    gl.useProgram(this.program);
    this.buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW,
    );
    const attribute = gl.getAttribLocation(this.program, 'position');
    gl.enableVertexAttribArray(attribute);
    gl.vertexAttribPointer(attribute, 2, gl.FLOAT, false, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }
  draw(
    context: CanvasRenderingContext2D,
    start: P,
    end: P,
    width: number,
    height: number,
    dpr: number,
    radius = 5,
  ) {
    const gl = this.gl,
      w = Math.round(width * dpr),
      h = Math.round(height * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const uniform = (name: string) => gl.getUniformLocation(this.program, name);
    gl.uniform2f(uniform('extent'), w, h);
    gl.uniform2f(uniform('v_a'), start[0] * dpr, start[1] * dpr);
    gl.uniform2f(uniform('v_b'), end[0] * dpr, end[1] * dpr);
    gl.uniform4f(uniform('v_core_color'), 0.98, 0.98, 0.99, 1);
    gl.uniform4f(uniform('v_edge_color'), 0.68, 0.71, 0.76, 1);
    gl.uniform1f(uniform('v_width'), 2 * dpr);
    gl.uniform1f(uniform('v_radius'), radius * dpr);
    gl.uniform1f(uniform('v_edge'), 0.75 * dpr);
    gl.uniform1f(uniform('v_smoothing'), 0.618);
    const pad = 10 * dpr,
      x = Math.max(0, Math.floor(Math.min(start[0], end[0]) * dpr - pad)),
      y = Math.max(0, Math.floor(h - Math.max(start[1], end[1]) * dpr - pad)),
      rw = Math.min(w - x, Math.ceil(Math.abs(end[0] - start[0]) * dpr + 2 * pad)),
      rh = Math.min(h - y, Math.ceil(Math.abs(end[1] - start[1]) * dpr + 2 * pad));
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x, y, rw, rh);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disable(gl.SCISSOR_TEST);
    context.globalAlpha = 1;
    context.drawImage(this.canvas, 0, 0, width, height);
  }
  dispose() {
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteProgram(this.program);
    this.gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
