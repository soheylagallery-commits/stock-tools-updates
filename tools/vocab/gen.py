# Builds tools/vocab/phrases.txt: concrete, photographable English nouns (WordNet, filtered by how common they are) + curated stock subjects.
import re, nltk
nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency as Z
ROOTS = {"artifact.n.01": 2.4, "food.n.01": 2.2, "food.n.02": 2.2, "plant.n.02": 2.4, "animal.n.01": 2.4, "natural_object.n.01": 2.4,
         "geological_formation.n.01": 2.2, "body_of_water.n.01": 2.2, "structure.n.01": 2.2, "person.n.01": 3.3, "substance.n.01": 3.0,
         "material.n.01": 2.6, "plant_part.n.01": 2.4, "fungus.n.01": 2.6, "weather.n.01": 2.6, "atmospheric_phenomenon.n.01": 2.6, "landscape.n.01": 2.4}
BAD_DOMAINS = set()
for n in ("derogation.n.01", "obscenity.n.01", "vulgarism.n.01", "slang.n.02", "ethnic_slur.n.01"):
    try: BAD_DOMAINS.add(wn.synset(n))
    except Exception: pass
BLOCK = set("""nigger negro jap chink spic kike faggot fag dyke retard whore slut bitch cunt dick cock prick pussy twat bastard tranny gook wetback
cracker honky gypsy midget cripple spastic slave hooker prostitute pimp stripper nazi terrorist bomb gun rifle pistol grenade cocaine heroin
weapon ammunition corpse coffin skull""".split())
def ok_word(w):
    return re.fullmatch(r"[a-z][a-z \-]{1,30}", w) and not any(b in w.split() for b in BLOCK)
out = {}
for root, thr in ROOTS.items():
    r = wn.synset(root)
    for s in r.closure(lambda x: x.hyponyms() + x.instance_hyponyms()):
        if any(d in BAD_DOMAINS for d in s.usage_domains()): continue
        if s.instance_hypernyms(): continue  # proper names (cities, people)
        for l in s.lemmas():
            w = l.name().replace("_", " ")
            if w != w.lower() or not ok_word(w): continue
            z = Z(w, "en")
            need = thr + 0.5
            if " " in w or "-" in w: need = max(need, 2.8)
            if s.lexname() in ("noun.plant", "noun.animal"): need = max(need, 3.3)
            if z >= need: out[w] = max(out.get(w, 0), z)
cur = [l.strip() for l in open("tools/vocab/curated.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]
words = sorted(out, key=lambda w: -out[w])[:12000]
P = list(dict.fromkeys(cur + words))
open("tools/vocab/phrases.txt", "w", encoding="utf-8").write("\n".join(P) + "\n")
open("tools/vocab/curated_count.txt", "w").write(str(len(dict.fromkeys(cur))))
print(len(cur), len(out), len(P))
