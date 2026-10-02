import os, json, wave, shutil
import numpy as np

D = os.path.expanduser("~/demo/data/speech")
OUT = os.path.join(D, "processed")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "samples"), exist_ok=True)

WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "marvin"]
N = 16000  # 1 second at 16 kHz

def read_set(name):
    with open(os.path.join(D, name)) as f:
        return set(line.strip() for line in f if line.strip())

val_set = read_set("validation_list.txt")
test_set = read_set("testing_list.txt")

# collect file lists per split, sorted so everything is deterministic
items = {"train": [], "val": [], "test": []}
for label, w in enumerate(WORDS):
    for fn in sorted(os.listdir(os.path.join(D, w))):
        if not fn.endswith(".wav"):
            continue
        rel = f"{w}/{fn}"
        split = "val" if rel in val_set else "test" if rel in test_set else "train"
        items[split].append((rel, label))

problems = {"rate": 0, "channels": 0, "width": 0, "padded": 0, "trimmed": 0}
for split, lst in items.items():
    X = np.zeros((len(lst), N), dtype=np.int16)
    y = np.zeros(len(lst), dtype=np.int8)
    speakers = []
    for i, (rel, label) in enumerate(lst):
        with wave.open(os.path.join(D, rel)) as w:
            if w.getframerate() != 16000: problems["rate"] += 1
            if w.getnchannels() != 1: problems["channels"] += 1
            if w.getsampwidth() != 2: problems["width"] += 1
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if len(a) < N:
            problems["padded"] += 1
            X[i, :len(a)] = a            # zero-pad at the end
        else:
            if len(a) > N: problems["trimmed"] += 1
            X[i] = a[:N]
        y[i] = label
        speakers.append(rel.split("/")[1].split("_")[0])
    np.save(os.path.join(OUT, f"X_{split}.npy"), X)
    np.save(os.path.join(OUT, f"y_{split}.npy"), y)
    with open(os.path.join(OUT, f"files_{split}.json"), "w") as f:
        json.dump([r for r, _ in lst], f)
    with open(os.path.join(OUT, f"speakers_{split}.json"), "w") as f:
        json.dump(speakers, f)
    per = [int((y == k).sum()) for k in range(len(WORDS))]
    print(split, len(lst), per)

# small fixed sample set for the repo: 2 clips per word (not marvin) from the test split
samples = []
for label, w in enumerate(WORDS[:10]):
    picked = [rel for rel, l in items["test"] if l == label][:2]
    for rel in picked:
        dst = os.path.join(OUT, "samples", rel.replace("/", "__"))
        shutil.copy(os.path.join(D, rel), dst)
        samples.append({"file": os.path.basename(dst), "label": label, "word": w})
json.dump(samples, open(os.path.join(OUT, "samples.json"), "w"))
json.dump(WORDS, open(os.path.join(OUT, "classes.json"), "w"))

print("problems:", problems)
print("samples:", len(samples))
