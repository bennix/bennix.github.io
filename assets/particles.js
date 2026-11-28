import * as THREE from './three/three.module.min.js';

const tau = 2 * Math.PI;
const shapes = {
  hero(t, i) {
    const y = 2 * t - 1, r = Math.sqrt(1 - y * y), a = i * 2.39996;
    return [r * Math.cos(a), y, r * Math.sin(a)];
  },
  books(t, i) {
    const x = 2 * t - 1, leaf = (i % 11) / 10;
    return [x, Math.abs(x) * (.22 + leaf * .48) - .3, ((i % 23) / 22 - .5) * 1.1];
  },
  foundation(t, i) {
    return [((i % 19) / 9 - 1) * .92, (Math.floor(i / 19) % 3 - 1) * .36, (Math.floor(i / 57) / 4 - 1) * .65];
  },
  suite(t, i) {
    const a = t * tau * 3, b = (i % 3) * tau / 3;
    return [.58 * Math.cos(b) + .42 * Math.cos(a), .58 * Math.sin(b) + .42 * Math.sin(a), .35 * Math.sin(a + b)];
  },
  teaching(t, i) {
    const a = t * tau * 2 + (i % 2) * Math.PI;
    return [t * 2 - 1, .48 * Math.sin(a), .48 * Math.cos(a)];
  },
  research(t, i) {
    const a = t * tau, b = (i % 3) * Math.PI / 3;
    return [Math.cos(a) * Math.cos(b) - .25 * Math.sin(a) * Math.sin(b), Math.cos(a) * Math.sin(b) + .25 * Math.sin(a) * Math.cos(b), .65 * Math.sin(a)];
  },
  tools(t, i) {
    const a = t * tau, r = .78 + .12 * Math.sign(Math.cos(12 * a));
    return [r * Math.cos(a), r * Math.sin(a), (i % 3 - 1) * .1];
  },
  closing(t, i) {
    const a = t * tau + (i % 3) * tau / 3;
    return [t * Math.cos(a), t * Math.sin(a), .5 - t];
  }
};

function mount() {
  const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
  renderer.setSize(480, 320, false);
  const camera = new THREE.PerspectiveCamera(38, 1.5, .1, 10);
  camera.position.z = 3.6;
  const visible = new Set();
  const scenes = new Map();
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let frame = 0, last = -100;
  for (const section of document.querySelectorAll('main section')) {
    const shape = shapes[section.id || 'closing'];
    if (!shape) continue;
    const canvas = document.createElement('canvas');
    canvas.className = 'particles';
    canvas.width = 480;
    canvas.height = 320;
    canvas.setAttribute('aria-hidden','true');
    section.append(canvas);
    const positions = new Float32Array(513 * 3);
    for (let i = 0; i < 513; i++) positions.set(shape(i / 512, i), i * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const points = new THREE.Points(geometry, new THREE.PointsMaterial({color:0xe4bc75, size:.017}));
    const scene = new THREE.Scene();
    scene.add(points);
    scenes.set(canvas, {scene, points, ctx:canvas.getContext('2d')});
  }
  function draw(time) {
    frame = 0;
    if (document.hidden || !visible.size) return;
    if (time - last >= 40 || motion.matches) {
      last = time;
      for (const canvas of visible) {
        const {scene, points, ctx} = scenes.get(canvas);
        points.rotation.y = motion.matches ? .28 : .45 * Math.sin(time * .00022);
        points.rotation.x = .14;
        renderer.render(scene, camera);
        ctx.clearRect(0, 0, 480, 320);
        ctx.drawImage(renderer.domElement, 0, 0);
      }
    }
    if (!motion.matches) schedule();
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(draw); }
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) visible.add(entry.target);
      else visible.delete(entry.target);
    }
    schedule();
  });
  scenes.forEach((_, canvas) => observer.observe(canvas));
  motion.addEventListener('change', schedule);
  document.addEventListener('visibilitychange', schedule);
}
try { mount(); } catch (error) { console.warn('Particle rendering unavailable', error); }
