(() => {
  const canvas = document.getElementById('scene')
  const note = document.getElementById('scene-note')
  const ctx = canvas.getContext('2d')
  const poseLayer = document.createElement('canvas')
  const poseContext = poseLayer.getContext('2d')
  const colors = [[137,237,83],[255,202,67],[94,168,246],[241,145,205]]
  let latest = {}, geometry = {}, yaw, pitch, distance, dragging = null
  let refreshing = false
  const vadd = (a,b) => a.map((value,i)=>value+b[i])
  const vsub = (a,b) => a.map((value,i)=>value-b[i])
  const scale = (a,s) => a.map(value=>value*s)
  const dot = (a,b) => a.reduce((sum,value,i)=>sum+value*b[i],0)
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
  const unit = a => scale(a,1 / (Math.hypot(...a) || 1))
  const rgba = (c,a=1) => 'rgba('+c.join(',')+','+a+')'
  const target = [0,0.035,0.38]
  const size = () => [canvas.clientWidth, canvas.clientHeight]

  function reset() { yaw=-0.64; pitch=0.44; distance=1.08; draw() }
  function projector(width,height) {
    const offset = [Math.sin(yaw)*Math.cos(pitch),-Math.sin(pitch),-Math.cos(yaw)*Math.cos(pitch)]
    const eye = vadd(target,scale(offset,distance)), forward=unit(vsub(target,eye))
    const right=unit(cross(forward,[0,-1,0])), up=cross(right,forward)
    const zoom=0.96*Math.min(width,height)
    return point => {
      const delta=vsub(point,eye), z=dot(delta,forward)
      return z > 0.02 ? [width*0.5+zoom*dot(delta,right)/z,height*0.50-zoom*dot(delta,up)/z,z] : null
    }
  }
  function polygon(points, stroke, fill, width=1, surface=ctx) {
    if (points.some(point=>!point)) return
    surface.beginPath()
    surface.moveTo(points[0][0],points[0][1])
    for(const point of points.slice(1)) surface.lineTo(point[0],point[1])
    if (fill) { surface.closePath(); surface.fillStyle=fill; surface.fill() }
    if (stroke) {surface.strokeStyle=stroke; surface.lineWidth=width; surface.stroke()}
  }
  function text(point, value, color, dx=0, dy=0, surface=ctx) {
    if (!point) return
    surface.font='12px system-ui,sans-serif'
    surface.fillStyle=color
    surface.fillText(value,point[0]+dx,point[1]+dy)
  }
  function grid(project) {
    for(let i=-4;i<=4;i++) {
      const color=i===0?'#3e5559':'#293b3e'
      polygon([project([i*0.1,0.15,0]),project([i*0.1,0.15,0.8])],color)
      polygon([project([-0.4,0.15,(i+4)*0.1]),project([0.4,0.15,(i+4)*0.1])],color)
    }
  }
  function frustum(project,k,imageWidth,imageHeight) {
    const depth=0.58, origin=project([0,0,0])
    const corners=[[0,0],[imageWidth,0],[imageWidth,imageHeight],[0,imageHeight]]
      .map(([x,y])=>project([(x-k[0][2])*depth/k[0][0],(y-k[1][2])*depth/k[1][1],depth]))
    polygon([...corners,corners[0]],'#75cfe8','rgba(107,195,216,.065)')
    for(const corner of corners) polygon([origin,corner],'#63a8ba')
    if (origin) {
      ctx.fillStyle='#9ee4f3'
      ctx.beginPath();ctx.arc(origin[0],origin[1],4,0,2*Math.PI);ctx.fill()
      text(origin,'D435','#bdebf5',7,-7)
    }
    const horizontal=(Math.atan(k[0][2]/k[0][0])+Math.atan((imageWidth-k[0][2])/k[0][0]))*180/Math.PI
    const vertical=(Math.atan(k[1][2]/k[1][1])+Math.atan((imageHeight-k[1][2])/k[1][1]))*180/Math.PI
    return horizontal.toFixed(1)+'° × '+vertical.toFixed(1)+'° camera FOV'
  }
  function meshes(project, poses, surface, offset=0) {
    const triangles=[], labels=[]
    for (const [index,pose] of poses.entries()) {
      const i=index+offset
      const color=colors[i%colors.length]
      const world=geometry.vertices.map(v=>pose.slice(0,3).map(row=>dot(row.slice(0,3),v)+row[3]))
      const projected=world.map(project)
      for (const face of geometry.faces) {
        const points=face.map(index=>projected[index])
        if(points.some(point=>!point)) continue
        const [a,b,c]=face.map(index=>world[index])
        const direction=unit(cross(vsub(b,a),vsub(c,a)))
        const light=0.44+0.38*Math.abs(dot(direction,[0.44,-0.66,-0.60]))
        const fill=rgba(color.map(channel=>Math.round(channel*light)))
        triangles.push({points,fill,depth:points.reduce((sum,p)=>sum+p[2]/3,0)})
      }
      const position=pose.slice(0,3).map(row=>row[3])
      labels.push({position:project(position),name:'Object '+(i+1),color:rgba(color)})
    }
    triangles.sort((a,b)=>b.depth-a.depth)
    for(const triangle of triangles) polygon(triangle.points,triangle.fill,triangle.fill,0.6,surface)
    for(const label of labels) text(label.position,label.name,label.color,8,-8,surface)
  }
  function markers(project, scene) {
    for(const [i,entry] of scene.markers.entries()) {
      if(!entry) continue
      const point=project(entry.position)
      if(!point) continue
      ctx.fillStyle=rgba(colors[i%colors.length])
      ctx.beginPath();ctx.arc(point[0],point[1],4,0,2*Math.PI);ctx.fill()
      text(point,'Depth '+(i+1),ctx.fillStyle,8,-7)
    }
  }
  function drawPoses(project, width, height, dpr) {
    const alpha = latest.pose_alpha ?? 1
    const solid = latest.solid_count ?? 0
    if (alpha === 1) {
      meshes(project, latest.poses, ctx)
      return
    }
    if (latest.poses.length === solid) {
      meshes(project, latest.poses, ctx)
      return
    }
    if (poseLayer.width !== canvas.width || poseLayer.height !== canvas.height) {
      poseLayer.width = canvas.width
      poseLayer.height = canvas.height
    }
    poseContext.setTransform(dpr, 0, 0, dpr, 0, 0)
    poseContext.clearRect(0, 0, width, height)
    meshes(project, latest.poses.slice(solid), poseContext, solid)
    ctx.save()
    ctx.globalAlpha = alpha
    ctx.drawImage(poseLayer, 0, 0, width, height)
    ctx.restore()
    if (solid) meshes(project, latest.poses.slice(0, solid), ctx)
  }
  function draw() {
    const [width,height]=size()
    if(!width || !height) return
    const dpr=Math.min(window.devicePixelRatio||1,2)
    const pixels=[Math.round(width*dpr),Math.round(height*dpr)]
    if(canvas.width!==pixels[0] || canvas.height!==pixels[1]) [canvas.width,canvas.height]=pixels
    ctx.setTransform(dpr,0,0,dpr,0,0)
    ctx.fillStyle='#0e1619'; ctx.fillRect(0,0,width,height)
    if(!latest.intrinsics) {
      ctx.font='14px system-ui,sans-serif'
      ctx.fillStyle='#a8bdbd';ctx.textAlign='center'
      ctx.fillText('Start the D435 stream to see the camera view.',width/2,height/2)
      ctx.textAlign='start'
      note.textContent='Drag to orbit · Scroll to zoom'
      return
    }
    const project=projector(width,height)
    grid(project)
    const calibration=frustum(project,latest.intrinsics,latest.width,latest.height)
    const showingDepth = !latest.poses.length && latest.markers.some(Boolean)
    if(latest.poses.length && geometry.id===latest.mesh) drawPoses(project,width,height,dpr)
    else if(showingDepth) markers(project,latest)
    note.textContent=calibration+(showingDepth?' · Detection depth estimates':'')+' · Drag to orbit · Scroll to zoom'
  }
  async function refresh() {
    if(!canvas.clientWidth || refreshing) return
    refreshing=true
    try {
      const response=await fetch('/scene.json',{cache:'no-store'})
      if(!response.ok) return
      latest=await response.json()
      if(latest.poses?.length && latest.mesh!==geometry.id) {
        const mesh=await fetch('/geometry.json',{cache:'no-store'})
        if(mesh.ok) geometry=await mesh.json()
      }
      draw()
    } catch {} finally {refreshing=false}
  }
  canvas.addEventListener('pointerdown',event=>{
    dragging=[event.clientX,event.clientY];canvas.setPointerCapture(event.pointerId)
  })
  canvas.addEventListener('pointermove',event=>{
    if(!dragging) return
    yaw+=(event.clientX-dragging[0])*0.008
    pitch=Math.max(-1.43,Math.min(1.43,pitch+(event.clientY-dragging[1])*0.008))
    dragging=[event.clientX,event.clientY];draw()
  })
  for(const type of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(type,()=>{dragging=null})
  canvas.addEventListener('wheel',event=>{
    event.preventDefault();distance=Math.max(.24,Math.min(3,distance*Math.exp(event.deltaY*.001)));draw()
  }, {passive:false})
  document.getElementById('reset-view').onclick=reset
  for(const button of document.querySelectorAll('[data-view]')) button.onclick=()=>{
    document.getElementById('viewports').dataset.layout=button.dataset.view
    for(const option of document.querySelectorAll('[data-view]')) option.setAttribute('aria-pressed',String(option===button))
    draw()
  }
  new ResizeObserver(draw).observe(canvas)
  reset();refresh()
  setInterval(refresh,125)
})()
