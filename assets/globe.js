import * as THREE from './three/three.module.min.js';

function mountGlobe() {
  const host = document.querySelector('.hero-bg');
  const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
  const canvas = renderer.domElement;
  canvas.className = 'globe-flow';
  host.append(canvas);
  const scene = new THREE.Scene();
  const globe = new THREE.Group();
  const orbits = new THREE.Group();
  const rig = new THREE.Group();
  scene.add(rig);
  rig.add(globe, orbits);
  rig.rotation.z = -.12;
  const camera = new THREE.OrthographicCamera(-2,2,1.2,-1.2,.1,20);
  camera.position.z = 5;
  const tau = Math.PI * 2;
  let seed = 391;
  function random() {
    seed = (Math.imul(seed,1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  const material = new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
    uniforms:{scale:{value:1}},
    vertexShader:`
      attribute vec2 spark;
      uniform float scale;
      varying float light;
      void main(){
        vec4 eye = modelViewMatrix * vec4(position,1.);
        gl_Position = projectionMatrix * eye;
        float front = smoothstep(-1.,1.,eye.z + 5.);
        light = spark.y * mix(.18,1.,front);
        gl_PointSize = max(1.,spark.x * scale * mix(.78,1.12,front));
      }`,
    fragmentShader:`
      varying float light;
      void main(){
        float r = length(gl_PointCoord - .5) * 2.;
        if(r >= 1.) discard;
        float core = exp(-5. * r*r);
        vec3 gold = mix(vec3(.96,.64,.30),vec3(1.,.94,.78),core);
        gl_FragColor = vec4(gold, min(1.,light * core) * (1.-r*r));
      }`
  });
  let xyz = [], spark = [];
  function point(x,y,z,size,light) {
    xyz.push(x,y,z);
    spark.push(size,light);
  }
  function finish(parent) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(xyz,3));
    geometry.setAttribute('spark',new THREE.Float32BufferAttribute(spark,2));
    parent.add(new THREE.Points(geometry,material));
    xyz = []; spark = [];
  }
  function surface(lat,lon,size,light,r=1) {
    const ring = r*Math.cos(lat);
    point(ring*Math.sin(lon),r*Math.sin(lat),ring*Math.cos(lon),size,light);
  }
  for(let j=1;j<28;j++) {
    const lat = (j/28-.5)*Math.PI;
    const count = Math.ceil(900*Math.cos(lat));
    for(let i=0;i<count;i++) surface(lat,tau*i/count,1.8,.50+random()*.48);
  }
  for(let j=0;j<40;j++) {
    for(let i=0;i<430;i++) surface((i/429-.5)*Math.PI,tau*j/40,1.7,.42+random()*.52);
  }
  // 不对称的旋臂和亮星随球体公转，让自转方向在第一眼就能辨认。
  for(let i=0;i<7800;i++) {
    const t=random(), arm=i%3;
    const lat=(t-.52)*2.55 + (random()-.5)*.16;
    const lon=arm*tau/3 + 2.7*t + .46*Math.sin(5*t) + (random()-.5)*.18;
    surface(lat,lon,1.7+random()*2.7,.46+random()*.66,.99+random()*.018);
  }
  for(let i=0;i<320;i++) surface(Math.asin(2*random()-1),random()*tau,3+random()*5,1.35);
  finish(globe);
  const rings=[];
  for(const [radius,tilt,roll] of [[1.28,1.31,.42],[1.14,1.13,-.24],[1.39,1.44,.18]]) {
    const ring=new THREE.Group();
    ring.rotation.order='ZXY';
    ring.rotation.x=tilt;
    ring.rotation.z=roll;
    orbits.add(ring);
    for(let i=0;i<1200;i++) {
      const a=tau*i/1200;
      point(radius*Math.cos(a),radius*Math.sin(a),0,1.9,.62);
    }
    finish(ring);
    const runners=new THREE.Group();
    ring.add(runners);
    for(let j=0;j<5;j++) {
      for(let i=0;i<36;i++) {
        const a=tau*j/5-i*.0028;
        point(radius*Math.cos(a),radius*Math.sin(a),0,1.8+6*Math.exp(-i/5),1.6-i/30);
      }
    }
    finish(runners);
    rings.push(runners);
  }
  let visible=false, frame=0,last=0,time=0;
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  function render(){
    globe.rotation.set(.16,time*.34,-.05);
    orbits.rotation.y=.24*Math.sin(time*.25);
    rings.forEach((ring,i)=>{ring.rotation.z=time*(.53+.11*i)+i;});
    renderer.render(scene,camera);
  }
  function draw(now){
    if(last) time+=(now-last)/1000;
    last=now;
    render();
    frame=requestAnimationFrame(draw);
  }
  function sync(){
    cancelAnimationFrame(frame);
    last=0;
    if(!visible || document.hidden) return;
    render();
    if(!motion.matches) frame=requestAnimationFrame(draw);
  }
  function resize(){
    const w=host.clientWidth,h=host.clientHeight;
    if(!w || !h) return;
    const radius=Math.min(h*.435,w*(w<700?.66:.30));
    const cx=w*(w<700?.69:.72),cy=h*.47;
    camera.left=-cx/radius; camera.right=(w-cx)/radius;
    camera.top=cy/radius; camera.bottom=(cy-h)/radius;
    camera.updateProjectionMatrix();
    const dpr=Math.min(devicePixelRatio,1.75);
    renderer.setPixelRatio(dpr);
    renderer.setSize(w,h,false);
    material.uniforms.scale.value=dpr*Math.max(.68,radius/330);
    render();
  }
  resize();
  new ResizeObserver(resize).observe(host);
  new IntersectionObserver(([e])=>{visible=e.isIntersecting;sync();}).observe(host);
  document.addEventListener('visibilitychange',sync);
  motion.addEventListener('change',sync);
}
mountGlobe();
