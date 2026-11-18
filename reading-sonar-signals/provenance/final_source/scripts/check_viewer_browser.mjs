// Exercise the offline UI against recorded case data; no scientific computation.
import {spawn} from "node:child_process";
import {mkdtemp,readFile,writeFile,rm} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join,resolve} from "node:path";
import {pathToFileURL} from "node:url";
const args=process.argv.slice(2),viewer=resolve(args.find(x=>!x.startsWith("--"))||"viewer");
const allowUnmapped=args.includes("--allow-unmapped"),profile=await mkdtemp(join(tmpdir(),"sonar-ui-check-"));
const browser=spawn("google-chrome",["--headless","--no-sandbox","--disable-gpu","--disable-background-networking",
  "--remote-debugging-address=127.0.0.1","--remote-debugging-port=0",`--user-data-dir=${profile}`,"--window-size=1440,1100",pathToFileURL(join(viewer,"index.html")).href],{stdio:["ignore","ignore","pipe"]});
let logs="",socket;
browser.stderr.on("data",value=>{logs=(logs+value).slice(-4000);});
const sleep=ms=>new Promise(done=>setTimeout(done,ms));
try{
  let port;
  for(let attempt=0;attempt<60;attempt++){try{port=(await readFile(join(profile,"DevToolsActivePort"),"utf8")).split("\n")[0];break;}catch(_){await sleep(100);}}
  if(!port)throw new Error("Chrome debugger did not start: "+logs);
  const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const target=targets.find(x=>x.type==="page");if(!target)throw new Error("No browser page target");
  socket=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((done,reject)=>{socket.addEventListener("open",done,{once:true});socket.addEventListener("error",reject,{once:true});});
  let counter=0;const pending=new Map(),exceptions=[];
  socket.addEventListener("message",event=>{const value=JSON.parse(event.data);if(value.id){const request=pending.get(value.id);pending.delete(value.id);if(value.error)request.reject(new Error(value.error.message));else request.resolve(value.result);}else if(value.method==="Runtime.exceptionThrown")exceptions.push(value.params.exceptionDetails);});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++counter;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
  const evaluate=async expression=>{const result=await call("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw new Error(JSON.stringify(result.exceptionDetails));return result.result.value;};
  await call("Runtime.enable");await call("Page.enable");
  for(let attempt=0;attempt<50;attempt++){if(await evaluate("document.readyState==='complete' && document.querySelector('#case-select').options.length>0"))break;await sleep(100);}
  const initial=await evaluate("({cases:bundle.cases.length,panels:panels.length,offline:location.protocol==='file:',scientificTransformsInBrowser:false,original:JSON.stringify(current.report)})");
  if(!initial.offline||!initial.cases)throw new Error("Offline case data did not initialize");
  if(!allowUnmapped&&initial.panels<2)throw new Error("Expected synchronized CUDA-exported LOFAR/DEMON panels");
  const result=await evaluate(`(async()=>{
    const checks=[];
    const slider=document.querySelector('#time-cursor');slider.value=String(current.time_s[0]+1);slider.dispatchEvent(new Event('input',{bubbles:true}));
    if(Math.abs(cursor-current.time_s[0]-1)>.02)throw new Error('Shared time cursor did not update');checks.push('shared cursor control');
    const first=document.querySelector('.component');if(first){first.click();if(!selected||!document.querySelector('#selected-json').textContent)throw new Error('Component click did not reveal provenance');checks.push('component selection and evidence record');
      const original=JSON.stringify(current.report),prior=notes.length;document.querySelector('#correction-reason').value='Isolated automated UI smoke; not a benchmark annotation';document.querySelector('#save-correction').click();
      if(notes.length!==prior+1||JSON.stringify(current.report)!==original)throw new Error('Separate correction changed frozen output or failed');checks.push('separate correction preserves frozen report');}
    if(panels.length){const canvas=panels[0].canvas,rect=canvas.getBoundingClientRect(),p=layout(canvas,panels[0].view);
      canvas.dispatchEvent(new PointerEvent('pointermove',{clientX:rect.left+p.x(current.time_s[0]+2),clientY:rect.top+p.box.y+20,bubbles:true}));
      if(Math.abs(cursor-current.time_s[0]-2)>.1)throw new Error('Canvas cursor did not synchronize');checks.push('canvas pointer links original coordinates');
      document.querySelector('#display-mode').value='contrast';document.querySelector('#display-mode').dispatchEvent(new Event('change'));checks.push('labelled contrast replay');}
    if(current.actions.length){document.querySelector('#action-select').value=String(current.actions.length-1);document.querySelector('#action-select').dispatchEvent(new Event('change'));if(!document.querySelector('#action-json').textContent)throw new Error('Action replay unavailable');checks.push('actual recorded action selector');}
    const audio=document.querySelector('#audio');let audioSeconds=null;if(current.audio){audio.load();await new Promise(resolve=>{audio.addEventListener('loadedmetadata',resolve,{once:true});audio.addEventListener('error',resolve,{once:true});setTimeout(resolve,1500);});audioSeconds=Number.isFinite(audio.duration)?audio.duration:null;}
    return {checks,audioSeconds,sourceScene:current.scene_id,panels:panels.length,offlineControlsIndependentOfGPU:document.querySelector('#reinspect').disabled};
  })()`);
  if(exceptions.length)throw new Error("Browser exceptions: "+JSON.stringify(exceptions));
  const screenshot=await call("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
  await writeFile(join(profile,"screen.png"),Buffer.from(screenshot.data,"base64"));
  const output={status:"passed",...result,coordinateMapping:initial.panels>=2?"tested":"not_yet_exported",runtimeExceptions:exceptions.length,script:"check_viewer_browser.mjs"};
  // This report is a software interaction check; it contains no new annotations.
  await writeFile(join(viewer,"browser_check.json"),JSON.stringify(output,null,2)+"\n");
  await writeFile(join(viewer,"browser_check.png"),Buffer.from(screenshot.data,"base64"));
  console.log(JSON.stringify(output));
}finally{if(socket)socket.close();browser.kill("SIGTERM");await sleep(300);await rm(profile,{recursive:true,force:true});}
