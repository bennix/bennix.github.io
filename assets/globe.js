import * as THREE from './three/three.module.min.js';

async function animateGlobe() {
  const image = document.querySelector('.hero-bg img');
  await image.decode();
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const width = image.naturalWidth, height = image.naturalHeight;
  // 原图坐标同时决定采样、球面深度和裁切，避免不同屏宽下光点偏离球体。
  const cx = width * .708, cy = height * .459, radius = width * .239;
  const sampler = document.createElement('canvas');
  sampler.width = width;
  sampler.height = height;
  const context = sampler.getContext('2d');
  context.drawImage(image, 0, 0);
  const pixels = context.getImageData(0, 0, width, height).data;
  const positions = [], colors = [];
  for (let y = 1; y < height; y += 4) {
    for (let x = 1; x < width; x += 4) {
      const dx = x - cx, dy = cy - y;
      const depth = radius * radius - dx * dx - dy * dy;
      if (depth <= 0) continue;
      const i = (y * width + x) * 4;
      if (pixels[i + 1] < 65) continue;
      positions.push(dx, dy, Math.sqrt(depth));
      colors.push(pixels[i] / 255, pixels[i + 1] / 255, pixels[i + 2] / 255);
    }
  }
  const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
  renderer.setSize(width, height, false);
  const canvas = renderer.domElement;
  canvas.className = 'globe-flow';
  canvas.setAttribute('aria-hidden', 'true');
  image.after(canvas);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const dust = new THREE.Points(geometry, new THREE.ShaderMaterial({
    vertexColors:true, transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
    uniforms:{time:{value:0}},
    vertexShader:`
      uniform float time;
      varying vec3 tint;
      void main() {
        float phase = time * .38 + position.y * .009 + position.z * .006;
        tint = color * (.55 + .45 * sin(phase));
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.);
        gl_PointSize = 2.2 + 2.5 * color.g;
      }`,
    fragmentShader:`
      varying vec3 tint;
      void main() {
        float glow = max(0., 1. - 2. * length(gl_PointCoord - .5));
        gl_FragColor = vec4(tint, glow * glow * .85);
      }`
  }));
  const scene = new THREE.Scene();
  scene.add(dust);
  const camera = new THREE.OrthographicCamera(-cx, width-cx, cy, cy-height, .1, 2500);
  camera.position.z = 1200;
  let visible = false, frame = 0, last = 0, elapsed = 0;
  function draw(now) {
    if (now-last >= 33) {
      elapsed += last ? Math.min(now-last,80) / 1000 : 0;
      last = now;
      dust.rotation.y = .105 * Math.sin(elapsed * .22);
      dust.material.uniforms.time.value = elapsed;
      renderer.render(scene,camera);
    }
    frame = requestAnimationFrame(draw);
  }
  function sync() {
    if (motion.matches) canvas.remove();
    else if (!canvas.isConnected) image.after(canvas);
    cancelAnimationFrame(frame);
    last = 0;
    if (visible && !document.hidden && !motion.matches) frame = requestAnimationFrame(draw);
  }
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync(); }).observe(image);
  motion.addEventListener('change',sync);
  document.addEventListener('visibilitychange',sync);
}
animateGlobe().catch(error => console.warn('Globe animation unavailable',error));
