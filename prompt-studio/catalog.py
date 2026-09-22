import pathlib,json,re,math
ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT.parent/'multisensor'
records=json.loads((DATA/'selection.json').read_text());day=json.loads((DATA/'synthetic-selection.json').read_text())
SOURCES={'daylight':{'sequence':day['sequences']['Clear'],'label':'Clear daylight'}}
for w,seq in records['sequences'].items():SOURCES[w.lower()]={'sequence':seq,'label':w.replace('_',' ')+' recording'}
GROUPS={'fog':{'fog','foggy','mist','misty','haze','hazy'},'rain':{'rain','rainy','raining','drizzle'},'snow':{'snow','snowy','snowing','snowfall'},'night':{'night','dark','darkness','nighttime'},'warm':{'warm','sunset','golden','dusk','evening'},'original':{'original','clear','daylight','day','normal','unchanged'}}
FILLER=set('a an the this that my video clip scene drive driving road street traffic city urban with and in at on during of to make create generate show render turn into it weather lighting exposure light low heavy dense thick mild moderate gentle soft bright morning afternoon overcast use using please very more less without no'.split())
def interpret(body):
    prompt=body.get('prompt','')
    if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=400:raise ValueError('Enter a weather or lighting prompt, up to 400 characters.')
    words=re.findall('[a-z]+',prompt.lower());allowed=FILLER|set().union(*GROUPS.values())
    unknown=sorted(set(words)-allowed)
    if unknown:raise ValueError('Unsupported prompt words: '+', '.join(unknown[:8])+'. Use fog, rain, snow, night, warm light, or original. Objects and routes cannot be changed.')
    effects=[]
    for effect,keys in GROUPS.items():
        hits=[i for i,w in enumerate(words) if w in keys]
        if any(not any(x in {'no','without'} for x in words[max(0,i-2):i]) for i in hits):effects.append(effect)
    if not effects:raise ValueError('Specify fog, rain, snow, night, warm light, or original.')
    if 'original' in effects and len(effects)>1:raise ValueError('Use original alone, or specify synthetic effects without clear/day/original.')
    strength=body.get('strength',.7);seed=body.get('seed',42);source=body.get('source','daylight')
    if isinstance(strength,bool) or not isinstance(strength,(int,float)) or not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Strength must be between 0 and 1.')
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=999999:raise ValueError('Seed must be an integer from 0 to 999999.')
    if source not in SOURCES:raise ValueError('Select a source from the available dataset clips.')
    scale=1.25 if set(words)&{'heavy','dense','thick','very'} else .6 if set(words)&{'mild','gentle','soft'} or 'light' in words and 'warm' not in effects else 1.
    return dict(prompt=prompt.strip(),effects=effects,strength=min(1.,strength*scale),seed=seed,source=source,sequence=SOURCES[source]['sequence'],revision=records['revision'])
