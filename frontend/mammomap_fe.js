import React, { useState, useRef, useEffect, useCallback, useReducer } from "react";

// ─── Design tokens ────────────────────────────────────────────────────────────
const T = {
  bg0:"#080b12", bg1:"#0d1018", bg2:"#121620", bg3:"#181d2b", bg4:"#1e2435",
  bdr:"#1f2840", bdr2:"#171c2c",
  txt:"#dde2f0", txt2:"#7a8aaa", txt3:"#48566e",
  teal:"#00d4aa", red:"#e84c6a", amber:"#f5a94a",
  purple:"#8b6de8", blue:"#3c8cdc", green:"#3cc878",
};
const SUB_C = {"Luminal A":"#3cc878","Luminal B":"#f5a94a","HER2-enriched":"#e84c6a","Triple-negative":"#8b6de8"};
const RISK_C = {Low:"#3cc878",Intermediate:"#f5a94a",High:"#e84c6a"};
const SEG = {
  tumor:     {c:"#e84c6a", f:"rgba(232,76,106,0.52)"},
  stroma:    {c:"#3c8cdc", f:"rgba(60,140,220,0.44)"},
  lymphocyte:{c:"#3cc878", f:"rgba(60,200,120,0.50)"},
  necrosis:  {c:"#f5a94a", f:"rgba(245,169,74,0.48)"},
};
const IHC_DAB = {HER2:"rgba(185,105,25,",Ki67:"rgba(115,50,205,",ER:"rgba(195,45,105,",PR:"rgba(25,160,140,"};
const GENES = ["ESR1","PGR","ERBB2","MKI67","TP53","BRCA1","CDH1","PIK3CA","CCND1","GATA3"];
const ROI_LABELS = {
  2:["Tumor core","Invasive front"],
  3:["Tumor core","Invasive front","Stroma"],
  4:["Tumor core","Invasive front","Stroma","Lymph node"],
  5:["Tumor core","Invasive front","Stroma","Lymph node","Necrosis focus"],
  6:["Tumor core","Invasive front","Stroma","Lymph node","Necrosis focus","Peritumoral"],
  7:["Tumor core","Invasive front","Stroma","Lymph node","Necrosis focus","Peritumoral","DCIS focus"],
};
const ROI_COMP = {
  "Tumor core":    [.58,.22,.12,.05],
  "Invasive front":[.38,.36,.22,.02],
  "Stroma":        [.10,.70,.15,.02],
  "Lymph node":    [.18,.25,.50,.03],
  "Necrosis focus":[.28,.22,.06,.40],
  "Peritumoral":   [.20,.55,.20,.03],
  "DCIS focus":    [.50,.28,.16,.04],
};

// ─── Seeded deterministic random ─────────────────────────────────────────────
const mkRng = seed => {
  let s = (seed * 1664525 + 1013904223) >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0xffffffff; };
};
const ri = (rng, a, b) => Math.floor(rng() * (b - a) + a);
const rf = (rng, a, b) => rng() * (b - a) + a;
const rnd = (v, d = 1) => Math.round(v * 10 ** d) / 10 ** d;

// ─── Per-ROI analysis generator ───────────────────────────────────────────────
function genAnalysis(pIdx, rIdx, subtype, label) {
  const rng = mkRng(pIdx * 1000 + rIdx * 37 + 91);
  const isLumA = subtype === "Luminal A", isLumB = subtype === "Luminal B";
  const isHER2 = subtype === "HER2-enriched", isTNBC = subtype === "Triple-negative";
  const base = {
    er:   isLumA?.80:isLumB?.55:isHER2?.07:.05,
    pr:   isLumA?.65:isLumB?.40:isHER2?.05:.04,
    her2: isHER2?.76:isLumB&&rng()>.5?.22:.04,
    ki67: isLumA?.08:isTNBC?.60:isLumB?.28:.35,
  };
  const ihcMaps = {};
  for (const [st, bv] of Object.entries({HER2:base.her2,Ki67:base.ki67,ER:base.er,PR:base.pr}))
    ihcMaps[st] = Array.from({length:144}, () => Math.max(0, Math.min(1, bv + rf(rng,-.22,.22))));
  const ihc = {
    HER2:{pct:rnd(base.her2*100+rf(rng,-7,7),1), grade:base.her2>.52?"3+":base.her2>.22?"2+":base.her2>.06?"1+":"0"},
    Ki67:{pct:rnd(base.ki67*100+rf(rng,-5,5),1), grade:`${rnd(base.ki67*100,1)}%`},
    ER:  {pct:rnd(base.er*100+rf(rng,-7,7),1),   grade:base.er>.62?"Allred 7/8":base.er>.32?"Allred 5/8":"Allred 2/8"},
    PR:  {pct:rnd(base.pr*100+rf(rng,-7,7),1),   grade:base.pr>.52?"Allred 6/8":"Allred 3/8"},
  };
  const comp = ROI_COMP[label] || [.35,.35,.20,.08];
  const segMask = Array.from({length:24}, () => {
    const roll = rng();
    const type = roll<comp[0]?"tumor":roll<comp[0]+comp[1]?"stroma":roll<comp[0]+comp[1]+comp[2]?"lymphocyte":"necrosis";
    return {type, cx:rf(rng,5,92), cy:rf(rng,5,92), rx:rf(rng,5,22), ry:rf(rng,4,16), angle:rf(rng,0,180)};
  });
  const spatial = Array.from({length:196}, (_,i) => {
    const cx=(i%14)/14*100, cy=Math.floor(i/14)/14*100;
    return {cx, cy, expr:Math.max(0, rf(rng,.08,.92) - Math.hypot(cx-50,cy-50)/85)};
  });
  const metrics = {
    tumor_purity: rnd((comp[0]+rf(rng,-.1,.12))*100,1),
    til_score:    rnd((comp[2]+rf(rng,-.06,.10))*100,1),
    stroma_ratio: rnd((comp[1]+rf(rng,-.1,.10))*100,1),
    necrosis_ratio:rnd((comp[3]+rf(rng,-.03,.05))*100,1),
  };
  return {ihcMaps, ihc, segMask, spatial, metrics};
}

// ─── 25-patient roster ────────────────────────────────────────────────────────
function buildRoster() {
  const dist=["Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal A","Luminal B","Luminal B","Luminal B","Luminal B","Luminal B","Luminal B","HER2-enriched","HER2-enriched","HER2-enriched","HER2-enriched","HER2-enriched","Triple-negative","Triple-negative","Triple-negative","Triple-negative"];
  const nrois=[3,4,2,5,7,3,6,2,4,3,4,2,5,3,6,4,2,3,5,7,4,3,2,4,5];
  return dist.map((subtype,i) => {
    const rng = mkRng(i*777+42);
    const n = nrois[i], risk = subtype==="Luminal A"?"Low":subtype==="Luminal B"?"Intermediate":"High";
    const age = ri(rng,38,72), grade = ri(rng,1,4);
    const het = rnd(({Luminal A:.15, "Luminal B":.33, "HER2-enriched":.50, "Triple-negative":.64}[subtype])+rf(rng,-.08,.14),2);
    const labels = ROI_LABELS[n];
    const rois = labels.map((label,j) => ({idx:j+1, label, analysis:genAnalysis(i,j,subtype,label)}));
    const aggIhc = {};
    for (const stain of["HER2","Ki67","ER","PR"]) {
      const pcts = rois.map(r=>r.analysis.ihc[stain].pct);
      aggIhc[stain] = {pct:rnd(pcts.reduce((s,v)=>s+v,0)/n,1), grade:rois[0].analysis.ihc[stain].grade, spread:rnd(Math.max(...pcts)-Math.min(...pcts),1)};
    }
    const aggMetrics = {
      tumor_purity: rnd(rois.reduce((s,r)=>s+r.analysis.metrics.tumor_purity,0)/n,1),
      til_score:    rnd(rois.reduce((s,r)=>s+r.analysis.metrics.til_score,0)/n,1),
      stroma_ratio: rnd(rois.reduce((s,r)=>s+r.analysis.metrics.stroma_ratio,0)/n,1),
      necrosis_ratio:rnd(rois.reduce((s,r)=>s+r.analysis.metrics.necrosis_ratio,0)/n,1),
    };
    return {id:`PAT-${String(i+1).padStart(3,"0")}`,code:`BRCA-2024-${String(i+1).padStart(3,"0")}`,age,subtype,risk,grade:`Grade ${grade}`,n_rois:n,het_score:het,rois,aggIhc,aggMetrics,seed:i};
  });
}
const PATIENTS = buildRoster();

function blendRois(rois) {
  const n = rois.length;
  const segMask = rois.flatMap(r=>r.analysis.segMask);
  const ihcMaps = {};
  for (const st of["HER2","Ki67","ER","PR"])
    ihcMaps[st] = Array.from({length:144},(_,i)=>rois.reduce((s,r)=>s+(r.analysis.ihcMaps[st][i]||0),0)/n);
  const spatial = rois[0].analysis.spatial.map((t,i)=>({...t,expr:rois.reduce((s,r)=>s+(r.analysis.spatial[i]?.expr||0),0)/n}));
  return {ihcMaps, segMask, spatial};
}

// ─── Canvas: simulated H&E background ────────────────────────────────────────
function drawHE(ctx, W, H, seed) {
  const rng = mkRng(seed * 17 + 3);
  const grad = ctx.createLinearGradient(0,0,W,H);
  grad.addColorStop(0,"#f5e4e8"); grad.addColorStop(1,"#eddee6");
  ctx.fillStyle = grad; ctx.fillRect(0,0,W,H);
  for (let i=0;i<20;i++) {
    ctx.beginPath(); ctx.moveTo(rng()*W,rng()*H);
    for(let j=0;j<4;j++) ctx.lineTo(rng()*W,rng()*H);
    ctx.strokeStyle=`rgba(210,155,165,${.2+rng()*.35})`; ctx.lineWidth=1+rng()*2.5; ctx.stroke();
  }
  for (let i=0;i<280;i++) {
    const x=rng()*W, y=rng()*H, rx=2+rng()*5, ry=rx*(.7+rng()*.6);
    ctx.beginPath(); ctx.ellipse(x,y,rx,ry,rng()*Math.PI,0,Math.PI*2);
    ctx.fillStyle=`rgba(55,28,88,${.35+rng()*.55})`; ctx.fill();
  }
  for (let i=0;i<40;i++) {
    ctx.beginPath(); ctx.arc(rng()*W,rng()*H,10+rng()*22,0,Math.PI*2);
    ctx.fillStyle=`rgba(240,175,185,${.15+rng()*.25})`; ctx.fill();
  }
}

// ─── Canvas overlay renderer ─────────────────────────────────────────────────
function renderOverlay({ctx, W, H, seed, groups, analysis}) {
  ctx.clearRect(0,0,W,H);
  const heL = groups.find(g=>g.id==="base")?.layers.find(l=>l.id==="he");
  if (heL?.vis) { ctx.save(); ctx.globalAlpha=heL.op/100; drawHE(ctx,W,H,seed); ctx.restore(); }

  if (!analysis) return;
  const {segMask, ihcMaps, spatial} = analysis;

  const segG = groups.find(g=>g.id==="seg");
  if (segG && segMask) {
    ctx.save(); ctx.globalCompositeOperation=segG.composite!=="normal"?segG.composite:"source-over";
    for (const l of segG.layers) {
      if (!l.vis) continue; ctx.globalAlpha=l.op/100;
      const col=SEG[l.cat];
      for (const r of segMask.filter(s=>s.type===l.cat)) {
        ctx.save(); ctx.translate(r.cx/100*W,r.cy/100*H); ctx.rotate(r.angle*Math.PI/180);
        ctx.beginPath(); ctx.ellipse(0,0,r.rx/100*W,r.ry/100*H,0,0,Math.PI*2);
        ctx.fillStyle=col.f; ctx.strokeStyle=col.c; ctx.lineWidth=1.2; ctx.fill(); ctx.stroke(); ctx.restore();
      }
    } ctx.restore();
  }

  const ihcG = groups.find(g=>g.id==="ihc");
  if (ihcG && ihcMaps) {
    ctx.save(); ctx.globalCompositeOperation=ihcG.composite!=="normal"?ihcG.composite:"source-over";
    for (const l of ihcG.layers) {
      if (!l.vis||!ihcMaps[l.stain]) continue; ctx.globalAlpha=l.op/100;
      const map=ihcMaps[l.stain], COLS=12, cw=W/COLS, ch=H/COLS, dab=IHC_DAB[l.stain];
      map.forEach((v,i)=>{ if(v<.1) return;
        const col=i%COLS,row=Math.floor(i/COLS),cx=col*cw,cy=row*ch;
        const g=ctx.createRadialGradient(cx+cw/2,cy+ch/2,0,cx+cw/2,cy+ch/2,Math.max(cw,ch)*.88);
        g.addColorStop(0,`${dab}${Math.min(.95,v*.9)})`); g.addColorStop(1,`${dab}0)`);
        ctx.fillStyle=g; ctx.fillRect(cx,cy,cw,ch);
      });
    } ctx.restore();
  }

  const spG=groups.find(g=>g.id==="spatial"), spL=spG?.layers.find(l=>l.id==="spatial_gene");
  if (spL?.vis && spatial) {
    ctx.save(); ctx.globalAlpha=spL.op/100;
    ctx.globalCompositeOperation=spG.composite!=="normal"?spG.composite:"source-over";
    for (const {cx,cy,expr} of spatial) {
      const px=cx/100*W, py=cy/100*H, r=W/30;
      const g=ctx.createRadialGradient(px,py,0,px,py,r*2.2);
      g.addColorStop(0,expr>.5?`rgba(240,100,30,${expr*.84})`:`rgba(25,75,200,${(1-expr)*.64})`);
      g.addColorStop(1,"transparent");
      ctx.beginPath(); ctx.arc(px,py,r*2.2,0,Math.PI*2); ctx.fillStyle=g; ctx.fill();
    } ctx.restore();
  }
}

// ─── Layer groups reducer ─────────────────────────────────────────────────────
const mkGroups = () => [
  {id:"base",  label:"Base Image",         expanded:true,  composite:"normal",   layers:[{id:"he",        label:"H&E",        vis:true,  op:100,color:T.txt,   locked:true,type:"he"}]},
  {id:"seg",   label:"Segmentation",       expanded:true,  composite:"normal",   layers:[{id:"seg_tumor",  label:"Tumor",      vis:true,  op:68, color:SEG.tumor.c,    type:"seg",cat:"tumor"},{id:"seg_stroma",label:"Stroma",vis:true,op:68,color:SEG.stroma.c,type:"seg",cat:"stroma"},{id:"seg_lymph",label:"Lymphocyte",vis:true,op:68,color:SEG.lymphocyte.c,type:"seg",cat:"lymphocyte"},{id:"seg_necr",label:"Necrosis",vis:false,op:68,color:SEG.necrosis.c,type:"seg",cat:"necrosis"}]},
  {id:"ihc",   label:"IHC Virtual Staining",expanded:true, composite:"multiply", layers:[{id:"ihc_her2",  label:"HER2",       vis:true,  op:60, color:"#c97a2a",type:"ihc",stain:"HER2"},{id:"ihc_ki67",label:"Ki67",vis:false,op:60,color:"#9b6de8",type:"ihc",stain:"Ki67"},{id:"ihc_er",label:"ER",vis:false,op:60,color:"#d44a8a",type:"ihc",stain:"ER"},{id:"ihc_pr",label:"PR",vis:false,op:60,color:"#2ac9b0",type:"ihc",stain:"PR"}]},
  {id:"spatial",label:"Spatial Expression",expanded:true,  composite:"screen",   layers:[{id:"spatial_gene",label:"ESR1",vis:true,op:55,color:null,type:"spatial",gene:"ESR1"}]},
];
function gRed(st,a) {
  const c=st.map(g=>({...g,layers:g.layers.map(l=>({...l}))}));
  if(a.type==="TG") return c.map(g=>g.id===a.gid?{...g,expanded:!g.expanded}:g);
  if(a.type==="TC") return c.map(g=>g.id===a.gid?{...g,composite:a.val}:g);
  if(a.type==="TL") return c.map(g=>({...g,layers:g.layers.map(l=>l.id===a.lid?{...l,vis:!l.vis}:l)}));
  if(a.type==="OP") return c.map(g=>({...g,layers:g.layers.map(l=>l.id===a.lid?{...l,op:a.val}:l)}));
  if(a.type==="GN") return c.map(g=>({...g,layers:g.layers.map(l=>l.id==="spatial_gene"?{...l,gene:a.val,label:a.val}:l)}));
  return c;
}

// ─── Tiny UI atoms ────────────────────────────────────────────────────────────
const Badge=({label,color,small})=>(<span style={{fontSize:small?9:10,fontWeight:600,padding:small?"1px 5px":"2px 8px",borderRadius:4,background:`${color}22`,color,letterSpacing:.3,flexShrink:0,whiteSpace:"nowrap"}}>{label}</span>);
const Eye=({on,toggle,color="#00d4aa"})=>(<div onClick={e=>{e.stopPropagation();toggle();}} title={on?"Hide":"Show"} style={{width:22,height:22,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer",color:on?color:T.txt3,flexShrink:0,borderRadius:4}} onMouseEnter={e=>e.currentTarget.style.background=T.bg4} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>
  {on?<svg width="14" height="10" viewBox="0 0 14 10" fill="none"><path d="M7 1C4 1 1.5 4 1.5 5S4 9 7 9 12.5 6 12.5 5 10 1 7 1Z" stroke="currentColor" strokeWidth="1.3" fill="none"/><circle cx="7" cy="5" r="2" fill="currentColor"/></svg>
    :<svg width="14" height="12" viewBox="0 0 14 12" fill="none"><path d="M1 1L13 11M7 2C4.5 2 2.5 4 1.5 5.5M7 9C9.5 9 11.5 7 12.5 5.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>}
</div>);
const MiniBar=({v,color})=>(<div style={{height:3,borderRadius:2,background:T.bdr,overflow:"hidden"}}><div style={{height:"100%",width:`${Math.min(100,Math.max(0,v*100))}%`,background:color,borderRadius:2,transition:"width .3s"}}/></div>);

// ─── Gallery thumbnail ─────────────────────────
function RoiThumb({seed, size=40, style={}}) {
  const ref = useRef();
  useEffect(()=>{
    const c=ref.current; if(!c)return;
    const ctx=c.getContext("2d");
    drawHE(ctx, size, size, seed);
  },[seed,size]);
  return <canvas ref={ref} width={size} height={size} style={{borderRadius:4,flexShrink:0,...style}}/>;
}

// ─── Patient gallery card ─────────────────────────────────────────────────────
function PatientCard({p, onSelect}) {
  const [hov, setHov] = useState(false);
  return(
    <div onClick={()=>onSelect(p)} onMouseEnter={()=>setHov(true)} onMouseLeave={()=>setHov(false)}
      style={{borderRadius:10,border:`1px solid ${hov?"#2a3555":T.bdr}`,background:hov?T.bg3:T.bg2,cursor:"pointer",overflow:"hidden",transition:"all .2s",transform:hov?"translateY(-2px)":"none",boxShadow:hov?"0 6px 24px rgba(0,0,0,.45)":"none",display:"flex",flexDirection:"column"}}>
      <div style={{position:"relative",height:110,background:T.bg0,overflow:"hidden"}}>
        <RoiThumb seed={p.seed} size={220} style={{width:"100%",height:"100%",borderRadius:0,objectFit:"cover"}}/>
        <div style={{position:"absolute",inset:0,background:"linear-gradient(to bottom,transparent 50%,rgba(8,11,18,.85))"}}/>
        <div style={{position:"absolute",top:7,left:7}}>
          <Badge label={p.subtype} color={SUB_C[p.subtype]}/>
        </div>
        <div style={{position:"absolute",top:7,right:7,display:"flex",gap:4}}>
          <span style={{fontSize:9,fontWeight:700,padding:"2px 6px",borderRadius:4,background:`${RISK_C[p.risk]}30`,color:RISK_C[p.risk],border:`1px solid ${RISK_C[p.risk]}44`}}>{p.risk}</span>
        </div>
        <div style={{position:"absolute",bottom:7,left:7,right:7,display:"flex",gap:3}}>
          {p.rois.slice(0,4).map((roi,j)=>(<RoiThumb key={j} seed={p.seed*10+j} size={22} style={{border:`1px solid rgba(255,255,255,.15)`,flexShrink:0}}/>))}
          {p.n_rois>4&&<div style={{width:22,height:22,borderRadius:4,background:"rgba(0,0,0,.5)",display:"flex",alignItems:"center",justifyContent:"center",fontSize:9,color:T.txt2,flexShrink:0}}>+{p.n_rois-4}</div>}
        </div>
      </div>
      <div style={{padding:"10px 12px",flex:1}}>
        <div style={{display:"flex",alignItems:"baseline",justifyContent:"space-between",marginBottom:3}}>
          <span style={{fontSize:12,fontWeight:700,color:T.txt}}>{p.code}</span>
          <span style={{fontSize:10,color:T.txt3}}>Age {p.age}</span>
        </div>
        <div style={{fontSize:10,color:T.txt2,marginBottom:8}}>{p.grade} · {p.n_rois} ROIs</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5,marginBottom:8}}>
          {[["Tumor",p.aggMetrics.tumor_purity+"%",T.red],["TIL",p.aggMetrics.til_score+"%",T.green]].map(([l,v,c])=>(
            <div key={l} style={{background:T.bg0,borderRadius:5,padding:"4px 7px"}}>
              <div style={{fontSize:9,color:T.txt3}}>{l}</div>
              <div style={{fontSize:12,fontWeight:700,color:c}}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <div style={{fontSize:10,color:T.txt3}}>ITH <span style={{fontWeight:600,color:p.het_score>.5?T.red:p.het_score>.3?T.amber:T.green}}>{p.het_score}</span></div>
          <span style={{fontSize:10,color:hov?T.teal:T.txt3,fontWeight:hov?600:400,transition:"color .15s"}}>Open →</span>
        </div>
      </div>
    </div>
  );
}

// ─── Patient gallery ──────────────────────────────────────────────────────────
function Gallery({onSelect}) {
  const [fSub, setFSub] = useState("all");
  const [fRisk, setFRisk] = useState("all");
  const [fROI, setFROI] = useState("all");
  const [search, setSearch] = useState("");

  const visible = PATIENTS.filter(p=>{
    if(fSub!=="all"&&p.subtype!==fSub) return false;
    if(fRisk!=="all"&&p.risk!==fRisk) return false;
    if(fROI==="2-3"&&p.n_rois>3) return false;
    if(fROI==="4-5"&&(p.n_rois<4||p.n_rois>5)) return false;
    if(fROI==="6-7"&&p.n_rois<6) return false;
    if(search&&!p.code.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const chip=(lbl,val,cur,set,col)=>(
    <button onClick={()=>set(v=>v===val?"all":val)} style={{padding:"3px 10px",borderRadius:12,border:`1px solid ${cur===val?(col||T.teal):T.bdr}`,background:cur===val?`${col||T.teal}1a`:"transparent",color:cur===val?(col||T.teal):T.txt2,fontSize:10,fontWeight:cur===val?600:400,cursor:"pointer",transition:"all .12s",whiteSpace:"nowrap"}}>
      {lbl}
    </button>
  );

  return(
    <div style={{background:T.bg1,minHeight:"100vh",fontFamily:"system-ui,-apple-system,sans-serif",color:T.txt}}>
      <div style={{borderBottom:`1px solid ${T.bdr}`,padding:"10px 24px",display:"flex",alignItems:"center",gap:10,background:T.bg2}}>
        <div style={{width:26,height:26,borderRadius:6,background:`linear-gradient(135deg,${T.red},${T.purple})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12}}>🔬</div>
        <span style={{fontWeight:700,fontSize:13,letterSpacing:"-.2px"}}>PathVision</span>
        <div style={{height:12,width:1,background:T.bdr}}/>
        <span style={{fontSize:11,color:T.txt2}}>BRCA Demo Cohort</span>
        <Badge label={`${PATIENTS.length} patients`} color={T.teal}/>
        <div style={{flex:1}}/>
        <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search patient code…"
          style={{padding:"5px 12px",borderRadius:7,border:`1px solid ${T.bdr}`,background:T.bg3,color:T.txt,fontSize:11,width:200,outline:"none"}}/>
        <button style={{padding:"5px 14px",borderRadius:7,background:`linear-gradient(135deg,${T.red},${T.purple})`,border:"none",color:"#fff",fontSize:11,fontWeight:600,cursor:"pointer"}}>+ New patient</button>
      </div>
      <div style={{borderBottom:`1px solid ${T.bdr}`,padding:"7px 24px",display:"flex",alignItems:"center",gap:5,flexWrap:"wrap",background:T.bg2}}>
        <span style={{fontSize:10,color:T.txt3,marginRight:2}}>Subtype</span>
        {chip("All","all",fSub,setFSub)}
        {["Luminal A","Luminal B","HER2-enriched","Triple-negative"].map(s=>chip(s,s,fSub,setFSub,SUB_C[s]))}
        <div style={{width:1,height:12,background:T.bdr,margin:"0 4px"}}/>
        <span style={{fontSize:10,color:T.txt3,marginRight:2}}>Risk</span>
        {["Low","Intermediate","High"].map(r=>chip(r,r,fRisk,setFRisk,RISK_C[r]))}
        <div style={{width:1,height:12,background:T.bdr,margin:"0 4px"}}/>
        <span style={{fontSize:10,color:T.txt3,marginRight:2}}>ROIs</span>
        {[["2–3","2-3"],["4–5","4-5"],["6–7","6-7"]].map(([l,v])=>chip(l,v,fROI,setFROI))}
        <div style={{flex:1}}/>
        <span style={{fontSize:10,color:T.txt3}}>{visible.length} shown</span>
      </div>
      <div style={{padding:"18px 24px",display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:13}}>
        {visible.map(p=><PatientCard key={p.id} p={p} onSelect={onSelect}/>)}
        {!visible.length&&<div style={{gridColumn:"1/-1",textAlign:"center",padding:"60px 0",color:T.txt3,fontSize:13}}>No patients match the current filters.</div>}
      </div>
    </div>
  );
}

// ─── Patient profile + TissUUmaps viewer ─────────────────────────────────────
function Profile({p, onBack}) {
  const [activeRoi, setActiveRoi] = useState("aggregate");
  const [groups, dispatch] = useReducer(gRed, null, mkGroups);
  const [selLayer, setSelLayer] = useState(null);
  const [panelTab, setPanelTab] = useState("layers");
  const [gene, setGene] = useState("ESR1");
  const [coords, setCoords] = useState({x:0,y:0});
  const tf = useRef({x:20,y:20,scale:.9});
  const dragging = useRef(null);
  const rafId = useRef(null);
  const cvs = useRef(), cont = useRef();

  const curRoi = activeRoi==="aggregate" ? null : p.rois[activeRoi];
  const curAnalysis = activeRoi==="aggregate" ? blendRois(p.rois) : curRoi?.analysis;
  const curSeed = activeRoi==="aggregate" ? p.seed : (p.seed*10+(activeRoi));
  const curIhc = activeRoi==="aggregate" ? p.aggIhc : curRoi?.analysis?.ihc;
  const curMetrics = activeRoi==="aggregate" ? p.aggMetrics : curRoi?.analysis?.metrics;

  const draw = useCallback(()=>{
    if(rafId.current) cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(()=>{
      const c=cvs.current; if(!c)return;
      const ctx=c.getContext("2d"), W=c.width, H=c.height;
      const {x,y,scale:sc}=tf.current;
      ctx.clearRect(0,0,W,H);
      ctx.save(); ctx.translate(x,y); ctx.scale(sc,sc);
      renderOverlay({ctx,W:W/sc,H:H/sc,seed:curSeed,groups,analysis:curAnalysis});
      ctx.restore();
    });
  },[groups, curAnalysis, curSeed]);

  useEffect(()=>{ draw(); },[draw]);

  useEffect(()=>{
    const ro=new ResizeObserver(e=>{
      const{width,height}=e[0].contentRect;
      if(cvs.current){cvs.current.width=width;cvs.current.height=height;}
      tf.current={x:20,y:20,scale:Math.min((cvs.current?.width||400)/440,(cvs.current?.height||400)/440)*.9};
      draw();
    });
    if(cont.current) ro.observe(cont.current);
    return()=>ro.disconnect();
  },[draw]);

  const onWheel=useCallback(e=>{
    e.preventDefault();
    const c=cvs.current; if(!c)return;
    const r=c.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,f=e.deltaY<0?1.12:.89;
    const{x,y,scale:sc}=tf.current;
    tf.current={x:mx-(mx-x)*f,y:my-(my-y)*f,scale:Math.max(.1,Math.min(20,sc*f))};
    draw();
  },[draw]);

  const onMD=e=>{ dragging.current={x:e.clientX,y:e.clientY,tx:tf.current.x,ty:tf.current.y}; };
  const onMM=useCallback(e=>{
    if(dragging.current){tf.current.x=dragging.current.tx+(e.clientX-dragging.current.x);tf.current.y=dragging.current.ty+(e.clientY-dragging.current.y);draw();}
    const c=cvs.current; if(c){const r=c.getBoundingClientRect();setCoords({x:Math.round((e.clientX-r.left-tf.current.x)/tf.current.scale),y:Math.round((e.clientY-r.top-tf.current.y)/tf.current.scale)});}
  },[draw]);

  const zoom=d=>{const{x,y,scale:sc}=tf.current,c=cvs.current;if(!c)return;const mx=c.width/2,my=c.height/2,f=d>0?1.2:.83;tf.current={x:mx-(mx-x)*f,y:my-(my-y)*f,scale:Math.max(.1,Math.min(20,sc*f))};draw();};
  const fit=()=>{const c=cvs.current;if(!c)return;const sc=Math.min(c.width/440,c.height/440)*.9;tf.current={x:(c.width-440*sc)/2,y:(c.height-440*sc)/2,scale:sc};draw();};

  const roiTabColor=idx=>idx==="aggregate"?T.teal:SUB_C[p.subtype];

  return(
    <div style={{display:"grid",gridTemplateColumns:"232px 1fr 272px",height:"100vh",background:T.bg1,fontFamily:"system-ui,-apple-system,sans-serif",color:T.txt,overflow:"hidden"}}>
      <div style={{borderRight:`1px solid ${T.bdr}`,display:"flex",flexDirection:"column",background:T.bg2,overflow:"hidden"}}>
        <div style={{padding:"11px 14px",borderBottom:`1px solid ${T.bdr}`,flexShrink:0}}>
          <button onClick={onBack} style={{display:"flex",alignItems:"center",gap:5,background:"transparent",border:"none",color:T.txt2,cursor:"pointer",fontSize:11,padding:"3px 0",marginBottom:10,width:"100%"}}>
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none"><path d="M7 2L3 5.5L7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
            Back to gallery
          </button>
          <div style={{fontSize:13,fontWeight:700,marginBottom:2}}>{p.code}</div>
          <div style={{fontSize:10,color:T.txt2,marginBottom:8}}>Age {p.age} · {p.grade} · Invasive ductal carcinoma</div>
          <div style={{display:"flex",gap:5,flexWrap:"wrap"}}>
            <Badge label={p.subtype} color={SUB_C[p.subtype]}/>
            <Badge label={`${p.risk} risk`} color={RISK_C[p.risk]}/>
          </div>
        </div>

        <div style={{padding:"10px 14px",borderBottom:`1px solid ${T.bdr}`,flexShrink:0}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:5}}>
            <span style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1}}>Intra-tumoral heterogeneity</span>
            <span style={{fontSize:15,fontWeight:700,color:p.het_score>.5?T.red:p.het_score>.3?T.amber:T.green}}>{p.het_score}</span>
          </div>
          <div style={{height:6,borderRadius:3,background:T.bdr,overflow:"hidden"}}>
            <div style={{height:"100%",width:`${p.het_score*100}%`,background:`linear-gradient(90deg,${T.green} 0%,${T.amber} 50%,${T.red} 100%)`,borderRadius:3}}/>
          </div>
          <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:T.txt3,marginTop:3}}><span>Low</span><span>High</span></div>
        </div>

        <div style={{padding:"10px 14px",borderBottom:`1px solid ${T.bdr}`,flexShrink:0}}>
          <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,marginBottom:8}}>
            IHC — {activeRoi==="aggregate"?"aggregate":curRoi?.label}
          </div>
          {Object.entries(curIhc||{}).map(([stain,d])=>{
            const sc={HER2:"#c97a2a",Ki67:"#9b6de8",ER:"#d44a8a",PR:"#2ac9b0"}[stain];
            return(<div key={stain} style={{marginBottom:7}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
                <span style={{fontWeight:600,color:sc}}>{stain}</span>
                <div style={{display:"flex",gap:5,alignItems:"center"}}>
                  {d.spread!=null&&d.spread>14&&<span title="High inter-ROI variability" style={{fontSize:9,color:T.amber}}>±{d.spread}%</span>}
                  <span style={{fontFamily:"monospace",color:T.txt,fontSize:10}}>{d.grade}</span>
                </div>
              </div>
              <MiniBar v={d.pct/100} color={sc}/>
            </div>);
          })}
        </div>

        <div style={{padding:"10px 14px",borderBottom:`1px solid ${T.bdr}`,flexShrink:0}}>
          <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,marginBottom:7}}>Tissue composition</div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:5}}>
            {[["Tumor",curMetrics?.tumor_purity+"%",T.red],["TIL",curMetrics?.til_score+"%",T.green],["Stroma",curMetrics?.stroma_ratio+"%",T.blue],["Necrosis",curMetrics?.necrosis_ratio+"%",T.amber]].map(([l,v,c])=>(
              <div key={l} style={{background:T.bg3,borderRadius:5,padding:"5px 8px"}}>
                <div style={{fontSize:9,color:T.txt3,marginBottom:1}}>{l}</div>
                <div style={{fontSize:14,fontWeight:700,color:c}}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{padding:"10px 14px",flex:1,overflowY:"auto"}}>
          <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,marginBottom:8}}>{p.n_rois} ROIs</div>
          <div onClick={()=>setActiveRoi("aggregate")} style={{display:"flex",alignItems:"center",gap:8,padding:"7px 9px",borderRadius:7,marginBottom:5,cursor:"pointer",background:activeRoi==="aggregate"?`${T.teal}18`:T.bg3,border:`1px solid ${activeRoi==="aggregate"?T.teal:T.bdr2}`,transition:"all .15s"}}>
            <div style={{width:26,height:26,borderRadius:4,background:`linear-gradient(135deg,${T.purple},${T.teal})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:12,flexShrink:0}}>⬡</div>
            <div style={{flex:1,minWidth:0}}>
              <div style={{fontSize:11,fontWeight:600,color:activeRoi==="aggregate"?T.teal:T.txt}}>Aggregate</div>
              <div style={{fontSize:9,color:T.txt3}}>All {p.n_rois} ROIs blended</div>
            </div>
          </div>
          {p.rois.map((roi,j)=>(
            <div key={j} onClick={()=>setActiveRoi(j)} style={{display:"flex",alignItems:"center",gap:8,padding:"6px 9px",borderRadius:7,marginBottom:4,cursor:"pointer",background:activeRoi===j?`${SUB_C[p.subtype]}18`:T.bg3,border:`1px solid ${activeRoi===j?SUB_C[p.subtype]:T.bdr2}`,transition:"all .15s"}}>
              <RoiThumb seed={p.seed*10+j} size={28} style={{flexShrink:0}}/>
              <div style={{flex:1,minWidth:0}}>
                <div style={{fontSize:11,fontWeight:600,color:activeRoi===j?SUB_C[p.subtype]:T.txt,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>ROI {roi.idx}</div>
                <div style={{fontSize:9,color:T.txt2,whiteSpace:"nowrap",overflow:"hidden",textOverflow:"ellipsis"}}>{roi.label}</div>
              </div>
              <div style={{fontSize:9,color:T.txt3}}>{roi.analysis.metrics.tumor_purity}%</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{display:"flex",flexDirection:"column",overflow:"hidden",background:T.bg1}}>
        <div style={{display:"flex",alignItems:"center",gap:6,padding:"7px 12px",borderBottom:`1px solid ${T.bdr}`,background:T.bg2,flexShrink:0}}>
          <span style={{fontSize:11,color:T.txt2,fontWeight:500}}>
            {activeRoi==="aggregate"?`Aggregate — all ${p.n_rois} ROIs`:`ROI ${activeRoi+1}: ${curRoi?.label}`}
          </span>
          <div style={{flex:1}}/>
          {[["⊕",()=>zoom(1),"Zoom in"],["⊖",()=>zoom(-1),"Zoom out"],["⊡",fit,"Fit"]].map(([ic,fn,tt])=>(
            <button key={tt} onClick={fn} title={tt} style={{padding:"3px 8px",borderRadius:4,border:`1px solid ${T.bdr}`,background:"transparent",color:T.txt2,fontSize:13,cursor:"pointer",transition:"background .1s"}}
              onMouseEnter={e=>e.currentTarget.style.background=T.bg4} onMouseLeave={e=>e.currentTarget.style.background="transparent"}>{ic}</button>
          ))}
          <span style={{fontSize:9,color:T.txt3,fontFamily:"monospace",marginLeft:2}}>{Math.round(tf.current.scale*100)}%</span>
        </div>

        <div style={{display:"flex",alignItems:"center",gap:3,padding:"5px 12px",borderBottom:`1px solid ${T.bdr}`,background:T.bg2,flexShrink:0,overflowX:"auto",scrollbarWidth:"none"}}>
          {[{id:"aggregate",label:"⬡ Aggregate"},...p.rois.map((r,j)=>({id:j,label:`ROI ${r.idx}: ${r.label}`}))].map(({id,label})=>(
            <button key={id} onClick={()=>setActiveRoi(id)} style={{padding:"4px 11px",borderRadius:6,border:`1px solid ${activeRoi===id?roiTabColor(id):T.bdr}`,background:activeRoi===id?`${roiTabColor(id)}1c`:"transparent",color:activeRoi===id?roiTabColor(id):T.txt2,fontSize:10,fontWeight:activeRoi===id?600:400,cursor:"pointer",flexShrink:0,whiteSpace:"nowrap",transition:"all .14s"}}>
              {label}
            </button>
          ))}
        </div>

        <div ref={cont} style={{flex:1,background:T.bg0,position:"relative",overflow:"hidden",cursor:"grab"}}
          onMouseDown={onMD} onMouseMove={onMM}
          onMouseUp={()=>{dragging.current=null;}} onMouseLeave={()=>{dragging.current=null;}}
          onWheel={onWheel}>
          <canvas ref={cvs} style={{display:"block",width:"100%",height:"100%"}}/>
        </div>

        <div style={{display:"flex",alignItems:"center",gap:12,padding:"3px 12px",borderTop:`1px solid ${T.bdr}`,background:T.bg2,flexShrink:0,fontSize:9,color:T.txt3,fontFamily:"monospace"}}>
          <span>x:{coords.x} y:{coords.y}</span>
          <div style={{width:1,height:8,background:T.bdr}}/>
          <span>scroll to zoom · drag to pan</span>
          <div style={{flex:1}}/>
          <div style={{display:"flex",gap:4,alignItems:"center"}}>
            <div style={{height:4,width:40,borderRadius:2,background:"linear-gradient(90deg,#1e50c8,#f0641e)"}}/>
            <span>spatial</span>
          </div>
        </div>
      </div>

      <div style={{borderLeft:`1px solid ${T.bdr}`,display:"flex",flexDirection:"column",background:T.bg2,overflow:"hidden"}}>
        <div style={{display:"flex",borderBottom:`1px solid ${T.bdr}`,padding:"6px 8px 0",background:T.bg1,flexShrink:0,gap:2}}>
          {[["layers","Layers"],["metrics","Metrics"]].map(([id,lbl])=>(
            <button key={id} onClick={()=>setPanelTab(id)} style={{fontSize:10,fontWeight:600,padding:"5px 10px",borderRadius:"4px 4px 0 0",cursor:"pointer",border:"none",background:panelTab===id?T.bg2:"transparent",color:panelTab===id?T.txt:T.txt2,borderBottom:panelTab===id?`2px solid ${T.teal}`:"2px solid transparent",transition:"all .14s"}}>{lbl}</button>
          ))}
        </div>

        {panelTab==="layers"&&(
          <div style={{flex:1,overflowY:"auto",padding:"7px 5px",display:"flex",flexDirection:"column",gap:3}}>
            <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,padding:"2px 7px",marginBottom:1}}>👁 toggle · click name to expand</div>
            {groups.map(g=>{
              const anyOn=g.layers.some(l=>l.vis), activeColor=g.layers.find(l=>l.vis)?.color||T.teal;
              return(<div key={g.id} style={{marginBottom:2}}>
                <div onClick={()=>dispatch({type:"TG",gid:g.id})} style={{display:"flex",alignItems:"center",gap:5,padding:"6px 8px",borderRadius:6,background:T.bg3,borderLeft:`2px solid ${anyOn?activeColor:T.bdr}`,cursor:"pointer",userSelect:"none"}}>
                  <svg width="9" height="9" viewBox="0 0 9 9" style={{transform:g.expanded?"rotate(90deg)":"none",transition:"transform .18s",flexShrink:0,color:T.txt3}} fill="none"><path d="M2 1.5L6.5 4.5L2 7.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/></svg>
                  <span style={{fontSize:11,fontWeight:600,color:T.txt,flex:1}}>{g.label}</span>
                  <select value={g.composite} onClick={e=>e.stopPropagation()} onChange={e=>{e.stopPropagation();dispatch({type:"TC",gid:g.id,val:e.target.value});}} style={{fontSize:9,padding:"1px 3px",borderRadius:3,border:`1px solid ${T.bdr}`,background:T.bg2,color:T.txt3,cursor:"pointer"}}>
                    {["normal","multiply","screen","overlay","lighter"].map(m=><option key={m} value={m}>{m}</option>)}
                  </select>
                  <Eye on={anyOn} toggle={()=>g.layers.forEach(l=>{if(anyOn===l.vis)dispatch({type:"TL",lid:l.id});})} color={activeColor}/>
                </div>
                {g.expanded&&g.layers.map(l=>(
                  <div key={l.id}>
                    <div onClick={()=>setSelLayer(selLayer===l.id?null:l.id)} style={{display:"flex",alignItems:"center",gap:5,padding:"5px 8px 5px 18px",cursor:"pointer",borderRadius:5,background:selLayer===l.id?T.bg4:"transparent",transition:"background .1s"}}>
                      <Eye on={l.vis} toggle={()=>dispatch({type:"TL",lid:l.id})} color={l.color||T.teal}/>
                      <div style={{width:8,height:8,borderRadius:2,background:l.vis?l.color||T.teal:T.txt3,flexShrink:0,transition:"background .14s"}}/>
                      <span style={{fontSize:11,color:l.vis?T.txt:T.txt3,flex:1,transition:"color .14s"}}>{l.label}</span>
                      {l.locked?<Badge label="base" color={T.txt3} small/>:<span style={{fontSize:9,color:T.txt3,fontFamily:"monospace"}}>{l.op}%</span>}
                    </div>
                    {selLayer===l.id&&!l.locked&&(
                      <div style={{padding:"5px 8px 7px 26px",display:"flex",flexDirection:"column",gap:6}}>
                        <div style={{display:"flex",alignItems:"center",gap:7}}>
                          <span style={{fontSize:9,color:T.txt3,width:40}}>Opacity</span>
                          <input type="range" min={0} max={100} value={l.op} onChange={e=>dispatch({type:"OP",lid:l.id,val:+e.target.value})} style={{flex:1,accentColor:l.color||T.teal}}/>
                          <span style={{fontSize:9,color:T.txt2,width:24,fontFamily:"monospace"}}>{l.op}%</span>
                        </div>
                        {l.type==="spatial"&&(
                          <div style={{display:"flex",alignItems:"center",gap:7}}>
                            <span style={{fontSize:9,color:T.txt3,width:40}}>Gene</span>
                            <select value={gene} onChange={e=>{setGene(e.target.value);dispatch({type:"GN",val:e.target.value});}} style={{flex:1,padding:"3px 6px",borderRadius:5,border:`1px solid ${T.bdr}`,background:T.bg3,color:T.txt,fontSize:10}}>
                              {GENES.map(g=><option key={g}>{g}</option>)}
                            </select>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>);
            })}
            <div style={{height:1,background:T.bdr,margin:"3px 0"}}/>
            <div style={{padding:"6px 9px",borderRadius:6,background:T.bg3,fontSize:9,color:T.txt2,lineHeight:1.7}}>
              <span style={{color:T.amber,fontWeight:600}}>IHC</span> Virchow2+ASP-NCE ·{" "}
              <span style={{color:T.red,fontWeight:600}}>Seg</span> Virchow2+ATM ·{" "}
              <span style={{color:T.purple,fontWeight:600}}>Spatial</span> SEQUOIA
            </div>
          </div>
        )}

        {panelTab==="metrics"&&(
          <div style={{flex:1,overflowY:"auto",padding:"9px 10px",display:"flex",flexDirection:"column",gap:9}}>
            <div style={{padding:"10px 12px",borderRadius:8,border:`1px solid ${RISK_C[p.risk]}44`,background:`${RISK_C[p.risk]}0d`,textAlign:"center"}}>
              <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,marginBottom:2}}>Case-level Risk</div>
              <div style={{fontSize:18,fontWeight:700,color:RISK_C[p.risk]}}>{p.risk}</div>
              <div style={{fontSize:9,color:T.txt3,marginTop:2}}>{p.subtype}</div>
            </div>
            <div style={{fontSize:9,color:T.txt3,textTransform:"uppercase",letterSpacing:1,fontWeight:600}}>Per-ROI overview</div>
            <div style={{overflowX:"auto"}}>
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
                <thead>
                  <tr>{["ROI","Label","Tumor%","TIL%","ER%","HER2"].map(h=><th key={h} style={{textAlign:"left",padding:"4px 6px",color:T.txt3,fontWeight:600,borderBottom:`1px solid ${T.bdr}`,whiteSpace:"nowrap"}}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {p.rois.map((roi,j)=>(
                    <tr key={j} onClick={()=>setActiveRoi(j)} style={{cursor:"pointer",background:activeRoi===j?`${SUB_C[p.subtype]}12`:"transparent",transition:"background .1s"}}>
                      <td style={{padding:"5px 6px",color:SUB_C[p.subtype],fontWeight:600}}>{roi.idx}</td>
                      <td style={{padding:"5px 6px",color:T.txt2,whiteSpace:"nowrap",maxWidth:80,overflow:"hidden",textOverflow:"ellipsis"}}>{roi.label}</td>
                      <td style={{padding:"5px 6px",color:T.red,fontFamily:"monospace"}}>{roi.analysis.metrics.tumor_purity}</td>
                      <td style={{padding:"5px 6px",color:T.green,fontFamily:"monospace"}}>{roi.analysis.metrics.til_score}</td>
                      <td style={{padding:"5px 6px",color:"#d44a8a",fontFamily:"monospace"}}>{roi.analysis.ihc.ER.pct}</td>
                      <td style={{padding:"5px 6px",color:"#c97a2a",fontFamily:"monospace"}}>{roi.analysis.ihc.HER2.grade}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {Object.entries(p.aggIhc).filter(([,d])=>d.spread>14).length>0&&(
              <div style={{padding:"8px 10px",borderRadius:7,border:`1px solid ${T.amber}44`,background:`${T.amber}0c`}}>
                <div style={{fontSize:9,color:T.amber,fontWeight:600,marginBottom:4}}>⚠ Inter-ROI discordance</div>
                {Object.entries(p.aggIhc).filter(([,d])=>d.spread>14).map(([stain,d])=>(
                  <div key={stain} style={{fontSize:10,color:T.txt2}}>{stain}: ±{d.spread}% variation across ROIs — verify with FISH</div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── App shell ────────────────────────────────────────────────────────────────
export default function App() {
  const [view, setView] = useState("gallery");
  const [patient, setPatient] = useState(null);
  const select = p => { setPatient(p); setView("profile"); };
  if (view==="profile"&&patient) return <Profile p={patient} onBack={()=>setView("gallery")}/>;
  return <Gallery onSelect={select}/>;
}