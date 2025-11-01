const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('file-input');
const form = document.getElementById('upload-form');
const analyzeBtn = document.getElementById('analyze-btn');
const previewWrap = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const results = document.getElementById('results');
const herbPicker = document.getElementById('herb-picker');
const useHerbBtn = document.getElementById('use-herb-btn');
const showMoreBtn = document.getElementById('show-more-btn');

let currentFile = null;
let allHerbs = [];

async function loadHerbs(){
  try{
    const res = await fetch('/api/herbs');
    if(!res.ok) return;
    const data = await res.json();
    allHerbs = data.items || [];
    // Populate select
    if(herbPicker){
      allHerbs.forEach(h => {
        const opt = document.createElement('option');
        opt.value = h.name;
        opt.textContent = h.name;
        herbPicker.appendChild(opt);
      });
    }
  }catch{}
}
loadHerbs();

function setFile(file){
  if(!file) return;
  currentFile = file;
  analyzeBtn.disabled = false;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewWrap.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

dropArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e)=> setFile(e.target.files[0]));

;['dragenter','dragover'].forEach(ev=>dropArea.addEventListener(ev, e=>{
  e.preventDefault();
  dropArea.style.background = '#f2f8f4';
}));
;['dragleave','drop'].forEach(ev=>dropArea.addEventListener(ev, e=>{
  e.preventDefault();
  dropArea.style.background = '';
}));

dropArea.addEventListener('drop', (e)=>{
  const file = e.dataTransfer.files[0];
  if(file) setFile(file);
});

async function renderPredictionsFromFile(k=5){
  if(!currentFile) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  results.classList.add('hidden');
  results.innerHTML = '';
  try{
    const fd = new FormData();
    fd.append('file', currentFile);
    const res = await fetch(`/api/predict?k=${encodeURIComponent(k)}`, {method:'POST', body: fd});
    if(!res.ok){
      const t = await res.text();
      throw new Error(t || 'Prediction failed');
    }
    const data = await res.json();
    const preds = data.predictions || [];
    if(preds.length === 0){
      results.innerHTML = '<div class="result">No herb recognized.</div>';
    } else {
      preds.forEach((p, i) => appendPredictionCard(p, i));
    }
    results.classList.remove('hidden');
  }catch(err){
    results.classList.remove('hidden');
    results.innerHTML = `<div class="result">${err.message}</div>`;
  }finally{
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
  }
}

function appendPredictionCard(p, i){
  const el = document.createElement('div');
  const pct = Math.round((p.confidence||0)*100);
  el.className = 'result';
  el.innerHTML = `
    <div class="name">#${(i!=null? i+1 : '')} ${p.name}</div>
    ${Number.isFinite(pct) ? `<div class=\"conf\">Confidence: ${pct}%</div>` : ''}
    ${p.scientific_name ? `<div class=\"conf\"><em>${p.scientific_name}</em></div>` : ''}
    ${p.matched_text ? `<div class=\"conf\">Matched: ${p.matched_text}</div>` : ''}
    <div class="details">${p.details || ''}</div>
    ${p.benefits ? `<div class=\"details\"><strong>Benefits:</strong> ${p.benefits}</div>` : ''}
    ${p.cautions ? `<div class=\"details\"><strong>Cautions:</strong> ${p.cautions}</div>` : ''}
    <div class="actions"><button class="btn secondary choose-btn" type="button">Use this herb</button></div>
  `;
  const choose = el.querySelector('.choose-btn');
  if (choose){
    choose.addEventListener('click', ()=>{
      results.innerHTML = '';
      el.classList.add('selected');
      const actions = el.querySelector('.actions');
      if(actions) actions.remove();
      results.appendChild(el);
    });
  }
  results.appendChild(el);
}

form.addEventListener('submit', async (e)=>{
  e.preventDefault();
  if(!currentFile) return;
  renderPredictionsFromFile(5);
});

if(showMoreBtn){
  showMoreBtn.addEventListener('click', ()=>{
    if(!currentFile){
      results.classList.remove('hidden');
      results.innerHTML = '<div class="result">Upload an image first.</div>';
      return;
    }
    renderPredictionsFromFile(10);
  });
}

if(useHerbBtn){
  useHerbBtn.addEventListener('click', ()=>{
    const name = herbPicker && herbPicker.value;
    if(!name){
      results.classList.remove('hidden');
      results.innerHTML = '<div class="result">Pick a herb from the list.</div>';
      return;
    }
    const h = allHerbs.find(x => x.name === name);
    if(!h){
      results.classList.remove('hidden');
      results.innerHTML = '<div class="result">Selected herb not found.</div>';
      return;
    }
    results.innerHTML = '';
    const payload = {
      name: h.name,
      scientific_name: h.scientific_name || '',
      details: h.details || '',
      benefits: h.benefits || '',
      cautions: h.cautions || '',
      matched_text: 'manual selection',
      confidence: NaN,
    };
    appendPredictionCard(payload, null);
    // Mark selected
    const last = results.lastElementChild;
    if(last){
      last.classList.add('selected');
      const actions = last.querySelector('.actions');
      if(actions) actions.remove();
    }
    results.classList.remove('hidden');
  });
}
