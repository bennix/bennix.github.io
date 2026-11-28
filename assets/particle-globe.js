(() => {
  const host = document.querySelector('.hero-bg');
  const canvas = document.createElement('canvas');
  canvas.className = 'globe-flow';
  const ctx = canvas.getContext('2d');
  host.append(canvas);
  const tau = Math.PI * 2;
  const surface = [];
  function add(lat, lon, size) {
    const ring = Math.cos(lat);
    surface.push([ring*Math.sin(lon), Math.sin(lat), ring*Math.cos(lon), size]);
  }
  for (let j=1;j<23;j++) {
    const lat=(j/23-.5)*Math.PI;
    const count=Math.ceil(270*Math.cos(lat));
    for(let i=0;i<count;i++) add(lat,i*tau/count,.85);
  }
  for(let j=0;j<32;j++) {
    for(let i=0;i<145;i++) add((i/144-.5)*Math.PI,j*tau/32,.82);
  }
  let seed=419;
  function random() {
    seed=(Math.imul(seed,1664525)+1013904223)>>>0;
    return seed/4294967296;
  }
  for(let i=0;i<3300;i++) {
    const t=random();
    add((t-.5)*2.6+(random()-.5)*.17,(i%3)*tau/3+3*t+.42*Math.sin(5*t)+(random()-.5)*.22,.8+random()*1.5);
  }
  for(let i=0;i<200;i++) add(Math.asin(2*random()-1),tau*random(),2.8);
  const rings=[[1.28,1.31,.42],[1.14,1.13,-.24],[1.39,1.44,.18]];
  let radius,cx,cy,width,height,elapsed=0,last=0,frame=0,visible=false;
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  const hero = host.closest('section');
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  let yaw = 0, pitch = .173, spin = 0;
  let targetX = 0, targetY = 0, viewX = 0, viewY = 0;
  let scroll = 0, drag = null;

  function updateScroll() {
    const progress = motion.matches ? 0 : clamp(-hero.getBoundingClientRect().top / height, 0, 1);
    scroll = progress * progress * (3 - 2 * progress);
    canvas.style.opacity = String(1 - scroll);
    canvas.style.transform = 'translateY(' + Math.round(height * scroll * .24) + 'px)';
  }
  function hover(event) {
    if (event.pointerType !== 'mouse') return;
    if (drag) {
      yaw = drag.yaw + (event.clientX - drag.x) / radius * 1.8;
      pitch = clamp(drag.pitch - (event.clientY - drag.y) / radius * 1.5, -1.2, 1.2);
      if (motion.matches) draw();
      return;
    }
    const rect = host.getBoundingClientRect();
    const x = (event.clientX - rect.left - cx) / radius;
    const y = (event.clientY - rect.top - cy) / radius;
    const over = x*x + y*y <= 1.15 && !event.target.closest('.hero-inner,a,button');
    hero.style.cursor = over ? 'grab' : '';
    targetX = over && !motion.matches ? x : 0;
    targetY = over && !motion.matches ? y : 0;
  }
  hero.addEventListener('pointermove', hover);
  hero.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0) return;
    hover(event);
    if (hero.style.cursor !== 'grab') return;
    event.preventDefault();
    drag = {x:event.clientX, y:event.clientY, yaw, pitch};
    hero.setPointerCapture(event.pointerId);
    hero.style.cursor = 'grabbing';
  });
  function resetPointer() {
    drag = null;
    targetX = targetY = 0;
    hero.style.cursor = '';
  }
  hero.addEventListener('lostpointercapture', resetPointer);
  hero.addEventListener('pointerleave', () => { if (!drag) resetPointer(); });
  window.addEventListener('scroll', updateScroll, {passive:true});
  function draw(){
    ctx.clearRect(0,0,width,height);
    const a=spin+yaw+viewX*.19+scroll*.48,c=Math.cos(a),s=Math.sin(a);
    const tilt=pitch+viewY*.14, ct=Math.cos(tilt), st=Math.sin(tilt);
    const rx=radius*(1-scroll*.13), ry=radius*(1-scroll*.46);
    const scale=Math.max(.75,radius/320);
    for(const [x,y,z,size] of surface){
      const depth=z*c-x*s;
      const px=x*c+z*s,py=y*ct-depth*st;
      ctx.fillStyle=depth<-.15?'#82633d':depth<.38?'#c99858':'#ffe4ae';
      const side=size*scale*(.82+.24*depth);
      ctx.fillRect(cx+rx*(px+.10*py),cy-ry*(py-.10*px),side,side);
    }
    rings.forEach(([r,tilt,roll],index)=>{
      const twist=roll+viewX*.06+scroll*.12+.10*Math.sin(elapsed*.26+index);
      const cosine=Math.cos(twist),sine=Math.sin(twist);
      for(let i=0;i<500;i++){
        const angle=i*tau/500+elapsed*(.48+.09*index);
        const x=r*Math.cos(angle),y=r*Math.sin(angle)*Math.cos(tilt);
        const px=cx+rx*(x*cosine-y*sine),py=cy-ry*(x*sine+y*cosine);
        const head=i%100===0;
        ctx.fillStyle=head?'#fff1cd':Math.sin(angle)>0?'#e8ba78':'#8f724a';
        const size=(head?4.7:i%100<7?2.3:1.3)*scale;
        ctx.fillRect(px-size/2,py-size/2,size,size);
      }
    });
  }
  function tick(now){
    if(!last || now-last>=32){
      const dt = last ? (now-last)/1000 : 0;
      elapsed += dt;
      if (!drag) spin += dt*.34;
      const blend = 1-Math.exp(-9*dt);
      viewX += (targetX-viewX)*blend;
      viewY += (targetY-viewY)*blend;
      last=now;
      draw();
    }
    frame=requestAnimationFrame(tick);
  }
  function sync(){
    cancelAnimationFrame(frame);
    last=0;
    if(!visible || document.hidden) return;
    draw();
    if(!motion.matches) frame=requestAnimationFrame(tick);
  }
  function resize(){
    width=host.clientWidth;height=host.clientHeight;
    const dpr=Math.min(devicePixelRatio,1.75);
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    ctx.setTransform(dpr,0,0,dpr,0,0);
    cx=width*(width<700?.69:.72);cy=height*.47;
    radius=Math.min(height*.435,width*(width<700?.66:.30));
    updateScroll();
    draw();
  }
  resize();
  new ResizeObserver(resize).observe(host);
  new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;sync();}).observe(host);
  document.addEventListener('visibilitychange',sync);
  motion.addEventListener('change', () => {
    viewX = viewY = targetX = targetY = 0;
    updateScroll();
    sync();
  });
})();
