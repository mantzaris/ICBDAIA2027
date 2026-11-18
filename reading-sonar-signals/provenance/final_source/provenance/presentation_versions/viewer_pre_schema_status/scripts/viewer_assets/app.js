"use strict";
// Offline rendering and interaction only. No scoring or signal transforms.
const bundle = JSON.parse(document.getElementById("case-data").textContent);
const el = id => document.getElementById(id);
const sources = {
  verified: {label:"Filtered report", color:"#ffb149", enabled:true},
  reference: {label:"Eligible AI reference", color:"#6ce8ff", enabled:true},
  raw: {label:"Raw report", color:"#e68fe9", enabled:false},
  measured: {label:"Measured proposals", color:"#98e092", enabled:false},
  disputed: {label:"Disputed AI", color:"#c1bccc", enabled:false},
  diagnostic4b: {label:"4B AI: unqualified", color:"#c2d7ff", enabled:false},
  diagnostic8b: {label:"8B AI: unqualified", color:"#e2b7ff", enabled:false},
  correction: {label:"Local correction", color:"#ff7589", enabled:false}
};
let current = null, components = [], selected = null, selectedInterval = null;
let cursor = 0, panels = [], imageCache = new Map(), frameRequested = false;
let serverReady = false, drag = null, notes = [];
const noteKey = "sonar-research-corrections-v1";
try { notes = JSON.parse(localStorage.getItem(noteKey) || "[]"); } catch (_) { notes = []; }
if(!Array.isArray(notes))notes=[];
function text(node, value) { node.textContent = value == null ? "" : String(value); }
function node(tag, content, className) { const n=document.createElement(tag); if(content!=null)text(n,content); if(className)n.className=className; return n; }
function finitePair(value) { return Array.isArray(value)&&value.length===2&&value.every(Number.isFinite); }
function geometry(event) { return event && finitePair(event.time_s)&&finitePair(event.frequency_hz)&&event.time_s[1]>event.time_s[0]&&event.frequency_hz[1]>=event.frequency_hz[0]; }
function overlap(a,b) { return a&&b&&Math.min(a[1],b[1])>Math.max(a[0],b[0]); }
function format(value) { return Number.isFinite(value) ? value.toFixed(2) : "unavailable"; }
function json(value) { return JSON.stringify(value,null,2); }
function refEligible(event, reference) {
  const types=reference?.eligible_types, domains=reference?.eligible_frequency_domains;
  return (!Array.isArray(types)||types.includes(event.type))&&(!Array.isArray(domains)||domains.includes(event.frequency_domain||"acoustic"));
}
function visibleComponents() { return components.filter(c=>sources[c.source].enabled); }
function collectComponents() {
  components=[];
  const append=(values,source)=>{ if(!Array.isArray(values))return; values.forEach((event,index)=>{
    if(event&&typeof event==="object")components.push({key:`${source}:${event.id||index}:${index}`,source,event});
  });};
  append(current.report?.components,"verified"); append(current.raw_report?.components,"raw");
  const reference=current.reference?.annotation||current.reference;
  append((reference?.events||[]).filter(e=>refEligible(e,reference)),"reference");
  append(current.measurement_components,"measured");
  const disputed=reference?.disputed_events;
  if(disputed&&typeof disputed==="object")Object.values(disputed).forEach(rows=>append(rows,"disputed"));
  for(const item of current.diagnostic_annotations||[])append(item.record.annotation?.events||[],item.annotator==="internvl4b"?"diagnostic4b":"diagnostic8b");
  append(notes.filter(n=>n.scene_id===current.scene_id).map(n=>n.corrected),"correction");
}
function requestDraw() { if(frameRequested)return; frameRequested=true; requestAnimationFrame(()=>{frameRequested=false;panels.forEach(drawPanel);}); }
function setCursor(value, seek=false) {
  cursor=Math.max(current.time_s[0],Math.min(current.time_s[1],value));
  el("time-cursor").value=String(cursor); text(el("time-label"),`${format(cursor)} s`);
  if(seek&&Number.isFinite(el("audio").duration))el("audio").currentTime=cursor-current.time_s[0];
  requestDraw();
}
function setIntervalSelection(interval) {
  selectedInterval=interval;
  el("inspect-start").value=format(interval[0]); el("inspect-end").value=format(interval[1]);
  if(!selected)text(el("selection-summary"),`Selected interval ${format(interval[0])}–${format(interval[1])} s. Shading links the same original-recording interval across views.`);
  renderComponents(); requestDraw();
}
function selectComponent(component) {
  selected=component;
  if(geometry(component.event)) { setIntervalSelection(component.event.time_s); setCursor(component.event.time_s[0]); }
  text(el("selected-json"),json(component.event));
  el("correction").value=json(component.event); el("correction-reason").value="";
  el("save-correction").disabled=false;
  const event=component.event, source=sources[component.source];
  text(el("selection-summary"),`${source.label}: ${event.id||"unnamed"} · ${event.type||"invalid morphology"}. ${geometry(event)?`${format(event.time_s[0])}–${format(event.time_s[1])} s; ${format(event.frequency_hz[0])}–${format(event.frequency_hz[1])} ${event.frequency_domain||"acoustic"} Hz.`:"Geometry is invalid or unavailable."}`);
  showEvidence(event.evidence_ids||[],component);
  renderComponents(); requestDraw();
}
function showEvidence(ids,component=null) {
  const container=el("evidence"); container.replaceChildren();
  const internal=!component||!["reference","disputed","diagnostic4b","diagnostic8b"].includes(component.source);
  const records=internal?(current.measurements||[]).filter(m=>ids.includes(m.id)):[];
  if(!records.length)container.append(node("p",component?.source==="reference"?"AI-reference provenance is shown in the raw record; it does not cite the controller's measurement graph.":"No cited internal measurement is available for this selection.","muted"));
  for(const record of records){
    const block=node("details",null,"evidence-entry"); block.open=true;
    block.append(node("summary",`${record.id} · ${record.tool||"measurement"}`),node("pre",json(record)));
    container.append(block);
  }
  if(component&&["verified","raw"].includes(component.source)){
    const audit=(current.verification_audit||[]).filter(a=>a.component_id===component.event.id);
    if(audit.length){container.append(node("p","Numerical verification record; not physical certification.","muted"),node("pre",json(audit)));}
  }
}
function renderComponents() {
  const container=el("components"); container.replaceChildren();
  for(const component of visibleComponents()){
    const event=component.event, button=node("button",null,"component"); button.type="button";
    if(selected?.key===component.key)button.classList.add("selected");
    else if(geometry(event)&&overlap(event.time_s,selectedInterval))button.classList.add("overlap");
    const color=node("strong",`${event.id||"unnamed"} · ${event.type||"invalid"}`);
    color.style.borderLeft=`4px solid ${sources[component.source].color}`; color.style.paddingLeft="6px";
    button.append(color,node("small",`${sources[component.source].label} · ${event.frequency_domain||"acoustic"}`));
    button.append(node("small",geometry(event)?`${format(event.time_s[0])}–${format(event.time_s[1])} s | ${format(event.frequency_hz[0])}–${format(event.frequency_hz[1])} Hz`:"Invalid or unavailable geometry"));
    if(event.carrier_band_hz)button.append(node("small",`Carrier ${event.carrier_band_hz.join("–")} Hz`));
    button.addEventListener("click",()=>selectComponent(component)); container.append(button);
  }
  if(!container.children.length)container.append(node("p","No recorded components in the enabled layers.","muted"));
}
function imageFor(url) {
  if(!imageCache.has(url)){const image=new Image();image.onload=requestDraw;image.src=url;imageCache.set(url,image);}
  return imageCache.get(url);
}
function layout(canvas,view) {
  const width=canvas.clientWidth,height=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);}
  const box={x:64,y:30,w:Math.max(30,width-80),h:Math.max(30,height-74)};
  const t=current.time_s,f=view.frequency_range_hz;
  return {width,height,dpr,box,x:v=>box.x+(v-t[0])/(t[1]-t[0])*box.w,y:v=>box.y+box.h-(v-f[0])/(f[1]-f[0])*box.h,
          time:x=>t[0]+(x-box.x)/box.w*(t[1]-t[0]),frequency:y=>f[0]+(box.y+box.h-y)/box.h*(f[1]-f[0])};
}
function belongs(event,view) {
  if((event.frequency_domain||"acoustic")!==view.frequency_domain)return false;
  if(view.frequency_domain==="modulation"&&JSON.stringify(event.carrier_band_hz)!==JSON.stringify(view.carrier_band_hz))return false;
  return geometry(event);
}
function drawPanel(panel) {
  const {canvas,view}=panel,ctx=canvas.getContext("2d"),p=layout(canvas,view),b=p.box;
  ctx.setTransform(p.dpr,0,0,p.dpr,0,0);ctx.clearRect(0,0,p.width,p.height);
  ctx.fillStyle="#fff";ctx.fillRect(0,0,p.width,p.height);ctx.fillStyle="#1b1522";ctx.fillRect(b.x,b.y,b.w,b.h);
  const mode=el("display-mode").value,raster=view.rasters[mode],image=imageFor(raster.url);
  ctx.save();ctx.beginPath();ctx.rect(b.x,b.y,b.w,b.h);ctx.clip();
  if(image.complete&&image.naturalWidth)ctx.drawImage(image,p.x(view.time_range_s[0]),b.y,p.x(view.time_range_s[1])-p.x(view.time_range_s[0]),b.h);
  if(selectedInterval){ctx.fillStyle="rgba(255,221,84,.17)";ctx.fillRect(p.x(selectedInterval[0]),b.y,p.x(selectedInterval[1])-p.x(selectedInterval[0]),b.h);}
  for(const component of visibleComponents()){
    const event=component.event;if(!belongs(event,view))continue;
    const active=selected?.key===component.key,coincident=overlap(event.time_s,selectedInterval),color=sources[component.source].color;
    ctx.globalAlpha=(selected&&!active&&!coincident) ? .34 : 1;ctx.strokeStyle=active?"#fff":color;ctx.lineWidth=active?3:coincident?2:1.3;
    ctx.setLineDash(component.source==="reference"?[6,4]:["disputed","diagnostic4b","diagnostic8b"].includes(component.source)?[2,4]:[]);
    const track=Array.isArray(event.frequency_track)&&event.frequency_track.every(v=>finitePair(v))?event.frequency_track:null;
    if(track&&track.length){ctx.beginPath();track.forEach((v,index)=>index?ctx.lineTo(p.x(v[0]),p.y(v[1])):ctx.moveTo(p.x(v[0]),p.y(v[1])));ctx.stroke();}
    else if(["tone","chirp","modulation"].includes(event.type)){const f=(event.frequency_hz[0]+event.frequency_hz[1])/2;ctx.beginPath();ctx.moveTo(p.x(event.time_s[0]),p.y(f));ctx.lineTo(p.x(event.time_s[1]),p.y(f));ctx.stroke();}
    else ctx.strokeRect(p.x(event.time_s[0]),p.y(event.frequency_hz[1]),p.x(event.time_s[1])-p.x(event.time_s[0]),p.y(event.frequency_hz[0])-p.y(event.frequency_hz[1]));
    if(component.source==="reference"&&Number.isFinite(event.frequency_uncertainty_hz)&&["tone","chirp","modulation"].includes(event.type)){
      const u=event.frequency_uncertainty_hz;ctx.fillStyle=color;ctx.globalAlpha=.10;
      const line=track&&track.length?track:[[event.time_s[0],(event.frequency_hz[0]+event.frequency_hz[1])/2],[event.time_s[1],(event.frequency_hz[0]+event.frequency_hz[1])/2]];
      ctx.beginPath();line.forEach((v,i)=>i?ctx.lineTo(p.x(v[0]),p.y(v[1]+u)):ctx.moveTo(p.x(v[0]),p.y(v[1]+u)));
      [...line].reverse().forEach(v=>ctx.lineTo(p.x(v[0]),p.y(v[1]-u)));ctx.closePath();ctx.fill();
    }
  }
  ctx.globalAlpha=1;ctx.setLineDash([]);ctx.strokeStyle="#000";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(p.x(cursor),b.y);ctx.lineTo(p.x(cursor),b.y+b.h);ctx.stroke();
  ctx.strokeStyle="#fff";ctx.lineWidth=1.2;ctx.stroke();ctx.restore();
  ctx.strokeStyle="#657787";ctx.lineWidth=1;ctx.strokeRect(b.x,b.y,b.w,b.h);ctx.fillStyle="#203647";ctx.font="11px system-ui";
  ctx.textAlign="center";
  for(let i=0;i<=4;i++){const t=current.time_s[0]+i/4*(current.time_s[1]-current.time_s[0]);ctx.fillText(t.toFixed(1),p.x(t),b.y+b.h+16);}
  ctx.fillText("Original-recording time (s)",b.x+b.w/2,p.height-5);
  ctx.textAlign="right";for(let i=0;i<=4;i++){const f=view.frequency_range_hz[0]+i/4*(view.frequency_range_hz[1]-view.frequency_range_hz[0]);ctx.fillText(f.toFixed(f<10?1:0),b.x-8,p.y(f)+4);}
  ctx.save();ctx.translate(13,b.y+b.h/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText(`${view.frequency_domain==="modulation"?"Modulation":"Acoustic"} frequency (Hz)`,0,0);ctx.restore();
  ctx.font="bold 12px system-ui";ctx.textAlign="left";ctx.fillText(view.name.replaceAll("_"," "),b.x,17);
}
function closestComponent(panel,x,y) {
  const p=layout(panel.canvas,panel.view),t=p.time(x);let closest=null,distance=9;
  for(const c of visibleComponents()){
    if(!belongs(c.event,panel.view)||t<c.event.time_s[0]||t>c.event.time_s[1])continue;
    const e=c.event;let d;
    if(!["tone","chirp","modulation"].includes(e.type))d=y>=p.y(e.frequency_hz[1])&&y<=p.y(e.frequency_hz[0])?4:Infinity;
    else if(Array.isArray(e.frequency_track)&&e.frequency_track.length>1){
      const points=e.frequency_track;let i=1;while(i<points.length-1&&points[i][0]<t)i++;
      const a=points[i-1],b=points[i],r=Math.max(0,Math.min(1,(t-a[0])/(b[0]-a[0]||1)));d=Math.abs(y-p.y(a[1]+r*(b[1]-a[1])));
    } else d=Math.abs(y-p.y((e.frequency_hz[0]+e.frequency_hz[1])/2));
    if(d<distance){distance=d;closest=c;}
  }return closest;
}
function addPanel(view,target=el("plots")) {
  const div=node("section",null,"plot"),canvas=node("canvas");canvas.setAttribute("aria-label",`${view.name} spectrogram; shared original-time cursor`);
  div.append(canvas);const meta=view.metadata||{};
  div.append(node("p",`${view.frequency_domain} Hz${view.carrier_band_hz?` · carrier ${view.carrier_band_hz.join("–")} acoustic Hz`:""} · window ${meta.window_s??"?"} s · native bin ${meta.bin_spacing_hz??"?"} Hz`));
  div.append(node("p",`Raw display ${view.rasters.raw.limits_db.map(v=>v.toFixed(1)).join(" to ")} dB. AI bands show stated uncertainty, not calibrated confidence.`));target.append(div);
  const panel={canvas,view};panels.push(panel);
  const pointer=e=>{const rect=canvas.getBoundingClientRect();return {x:e.clientX-rect.left,y:e.clientY-rect.top};};
  canvas.addEventListener("pointermove",e=>{const q=pointer(e),p=layout(canvas,view);if(q.x>=p.box.x&&q.x<=p.box.x+p.box.w)setCursor(p.time(q.x));if(drag?.panel===panel){const t=Math.max(current.time_s[0],Math.min(current.time_s[1],p.time(q.x)));setIntervalSelection([Math.min(drag.time,t),Math.max(drag.time,t)]);}});
  canvas.addEventListener("pointerdown",e=>{const q=pointer(e),p=layout(canvas,view);if(q.x<p.box.x||q.x>p.box.x+p.box.w)return;drag={panel,time:p.time(q.x),x:q.x};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener("pointerup",e=>{if(drag?.panel!==panel)return;const q=pointer(e),p=layout(canvas,view);if(Math.abs(q.x-drag.x)<5){const match=closestComponent(panel,q.x,q.y);if(match)selectComponent(match);else setCursor(p.time(q.x),true);}else{selected=null;el("save-correction").disabled=true;text(el("selected-json"),"");showEvidence([]);}drag=null;renderComponents();requestDraw();});
  canvas.addEventListener("pointercancel",()=>{drag=null;});new ResizeObserver(requestDraw).observe(canvas);requestDraw();
}
function chooseAction() {
  const index=Number(el("action-select").value),action=current.actions[index];const target=el("action-images");target.replaceChildren();
  if(!action){text(el("action-caption"),"No recorded inspection actions for this episode.");text(el("action-json"),"");return;}
  selected=null;el("save-correction").disabled=true;text(el("selected-json"),"");
  text(el("action-caption"),`Step ${action.step}: ${action.action?.name||"invalid action"} · ${action.status}. ${action.reason||""}`);text(el("action-json"),json(action));
  for(const item of action.saved_images||[]){const img=node("img");img.src=item.url;img.alt=`Unmodified recorded action ${action.step}`;target.append(img);}
  if(!target.children.length)target.append(node("p","This recorded call returned no image; its status and numeric result remain above.","muted"));
  if(finitePair(action.action?.time_s))setIntervalSelection(action.action.time_s);
  if(action.result?.id)showEvidence([action.result.id]);
}
function loadCase(index) {
  el("audio").pause();current=bundle.cases[index];selected=null;selectedInterval=null;panels=[];imageCache=new Map();collectComponents();
  text(el("case-summary"),`${current.scene_id} · ${current.domain} · ${current.method} · ${current.selection.purpose||"recorded case"}. Selection criterion met: ${String(current.selection.criterion_met)}.`);
  const reference=current.reference?.annotation||current.reference;
  sources.reference.label=Array.isArray(reference?.eligible_types)?"Eligible AI reference":"Legacy AI: unqualified";
  const referenceLabel=document.querySelector('[data-layer="reference"] .layer-label');if(referenceLabel)text(referenceLabel,sources.reference.label);
  text(el("reference-scope"),reference?`Status: ${reference.reference_status||reference.annotation_status||"recorded"}. Eligible types: ${JSON.stringify(reference.eligible_types??"legacy scope")}. Unqualified morphologies are unscored.`:"No AI-reference record supplied; these are model/measurement outputs only.");
  el("time-cursor").min=String(current.time_s[0]);el("time-cursor").max=String(current.time_s[1]);
  const audio=el("audio");audio.hidden=!current.audio;if(current.audio)audio.src=current.audio.url;else audio.removeAttribute("src");
  el("plots").replaceChildren();el("exploratory-views").replaceChildren();
  el("display-mode").disabled=!current.views.length;
  for(const view of current.views)addPanel(view);
  if(!current.views.length){el("plots").append(node("p","Development replay has recorded images only. Linked coordinate overlays require the GPU-exported view package; no image coordinates are guessed.","no-view"));for(const item of current.recorded_global_images){const img=node("img",null,"recorded-image");img.src=item.url;img.alt=item.name;el("plots").append(img);}}
  text(el("all-records"),json({reference:current.reference,unqualified_diagnostic_annotations:current.diagnostic_annotations,raw_report:current.raw_report,filtered_report:current.report,failure:current.failure}));
  text(el("selected-json"),"");text(el("selection-summary"),"Choose a component or interval.");el("evidence").replaceChildren();el("correction").value="";el("save-correction").disabled=true;text(el("correction-status"),"");
  const actionSelect=el("action-select");actionSelect.replaceChildren();current.actions.forEach((a,i)=>{const option=node("option",`${a.step}: ${a.action?.name||"invalid"} (${a.status})`);option.value=String(i);actionSelect.append(option);});actionSelect.disabled=!current.actions.length;
  setCursor(current.time_s[0]);setIntervalSelection([...current.time_s]);selectedInterval=null;chooseAction();
  if(!current.actions.length)text(el("selection-summary"),"Choose a component or interval.");
  displayExplanation();renderComponents();requestDraw();
}
Object.entries(sources).forEach(([key,source])=>{const label=node("label"),input=node("input");label.dataset.layer=key;input.type="checkbox";input.checked=source.enabled;input.addEventListener("change",()=>{source.enabled=input.checked;renderComponents();requestDraw();});label.append(input,node("span",source.label,"layer-label"));el("layers").append(label);});
bundle.cases.forEach((c,i)=>{const option=node("option",`${c.key} · ${c.scene_id} · ${c.domain}`);option.value=String(i);el("case-select").append(option);});
el("case-select").addEventListener("change",()=>loadCase(Number(el("case-select").value)));
el("time-cursor").addEventListener("input",()=>setCursor(Number(el("time-cursor").value),true));
el("audio").addEventListener("timeupdate",()=>{if(current)setCursor(current.time_s[0]+el("audio").currentTime);});
el("display-mode").addEventListener("change",()=>{displayExplanation();requestDraw();});
function displayExplanation(){text(el("display-explanation"),!current.views.length?"Original recorded images are replayed unchanged. Coordinate-linked raw/contrast views await the CUDA export.":el("display-mode").value==="raw"?"Raw PSD colors use a recorded 60 dB display range in uncalibrated digital units. No source level or range is implied.":"Display contrast subtracts each frame's frequency median on CUDA. Relative contrast can conceal level changes; it is not absolute PSD or a benchmark metric.");}
el("action-select").addEventListener("change",chooseAction);
el("clear-selection").addEventListener("click",()=>{selected=null;selectedInterval=null;text(el("selection-summary"),"Choose a component or interval.");text(el("selected-json"),"");el("evidence").replaceChildren();el("save-correction").disabled=true;renderComponents();requestDraw();});
el("save-correction").addEventListener("click",()=>{try{
  const corrected=JSON.parse(el("correction").value);if(!geometry(corrected))throw new Error("Correction needs finite ordered time_s and frequency_hz pairs.");
  if(!selected)throw new Error("Choose an original component first.");
  const entry={scene_id:current.scene_id,created_at:new Date().toISOString(),original_source:selected.source,original:selected.event,corrected,reason:el("correction-reason").value,scope:"local_exploratory_correction_excluded_from_frozen_benchmark"};
  notes.push(entry);try{localStorage.setItem(noteKey,json(notes));text(el("correction-status"),"Saved separately in this browser. Export to retain a portable copy.");}catch(_){text(el("correction-status"),"Browser storage is unavailable. Export now to retain this separate correction.");}
  collectComponents();renderComponents();requestDraw();
}catch(error){text(el("correction-status"),error.message);}});
el("export-notes").addEventListener("click",()=>{const blob=new Blob([json({scope:"exploratory_only",benchmark_modified:false,notes})],{type:"application/json"}),url=URL.createObjectURL(blob),link=node("a");link.href=url;link.download="separate-acoustic-corrections.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
el("reinspect").addEventListener("click",async()=>{
  if(!serverReady)return;el("reinspect").disabled=true;text(el("reinspect-status"),"Computing a new exploratory measurement on CUDA…");
  const action={name:el("inspect-kind").value,time_s:[Number(el("inspect-start").value),Number(el("inspect-end").value)],window_s:Number(el("inspect-window").value)};
  if(action.name==="inspect_modulation")action.carrier_band_hz=el("inspect-carrier").value.split(",").map(Number);else action.frequency_hz=[Number(el("inspect-low").value),Number(el("inspect-high").value)];
  try{const response=await fetch("/api/reinspect",{method:"POST",headers:{"Content-Type":"application/json"},body:json({scene_id:current.scene_id,action})});const data=await response.json();if(!response.ok)throw new Error(data.error||"Reinspection unavailable");
    text(el("reinspect-status"),`${data.result.status}: exploratory reinspection; excluded from frozen predictions and scores. ${data.remaining_calls} calls remain for this case.`);
    if(data.view){const note=node("p",`Exploratory call ${data.request_id}: ${action.name}; not an evaluated model action.`,"note");el("exploratory-views").append(note);addPanel(data.view,el("exploratory-views"));}
    const detail=node("details");detail.append(node("summary","Exploratory parameters, evidence, and provenance"),node("pre",json(data)));el("exploratory-views").append(detail);
  }catch(error){text(el("reinspect-status"),error.message);}finally{el("reinspect").disabled=!serverReady;}
});
async function detectServer(){if(!["http:","https:"].includes(location.protocol)||!["localhost","127.0.0.1","[::1]"].includes(location.hostname))return;try{const response=await fetch("/api/status"),data=await response.json();if(data.service==="sonar-local-cuda-viewer"&&data.cuda){serverReady=true;text(el("server-state"),"Local CUDA reinspection");text(el("reinspect-status"),`Local CUDA service available. At most ${data.max_calls_per_case} exploratory calls per case; frozen benchmark records remain unchanged.`);el("reinspect").disabled=false;}}catch(_){/* Offline replay needs no endpoint. */}}
if(bundle.cases.length){loadCase(0);displayExplanation();detectServer();}else{text(el("case-summary"),"No recorded cases in this package.");}
