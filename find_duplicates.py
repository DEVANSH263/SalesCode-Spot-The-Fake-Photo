"""Find duplicate and near-duplicate images in Data/real/."""
import hashlib
from pathlib import Path

from PIL import Image
import imagehash

REAL = Path(__file__).resolve().parent / "Data" / "real"
TH_NEAR = 6
TH_LOOSE = 10


def main():
    files = sorted(f for f in REAL.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Total images: {len(files)}\n")

    # Exact byte duplicates
    by_md5: dict[str, list[str]] = {}
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        by_md5.setdefault(h, []).append(f.name)

    exact = [names for names in by_md5.values() if len(names) > 1]
    print(f"=== EXACT DUPLICATES: {len(exact)} groups ===")
    for names in exact:
        print("  " + " | ".join(names))
    print()

    # Perceptual hashes
    hashes: dict[str, tuple] = {}
    for f in files:
        img = Image.open(f)
        hashes[f.name] = (imagehash.phash(img), img.size, f.stat().st_size)

    names = list(hashes.keys())
    phash_groups: list[list[str]] = []
    used: set[str] = set()

    for i, n1 in enumerate(names):
        if n1 in used:
            continue
        group = [n1]
        ph1 = hashes[n1][0]
        for n2 in names[i + 1 :]:
            if n2 in used:
                continue
            if ph1 - hashes[n2][0] <= TH_NEAR:
                group.append(n2)
        if len(group) > 1:
            used.update(group)
            phash_groups.append(group)

    print(f"=== NEAR-DUPLICATES (pHash distance <= {TH_NEAR}): {len(phash_groups)} groups ===")
    keep_near: list[str] = []
    remove_near: list[str] = []
    for gi, group in enumerate(phash_groups, 1):
        reps = sorted(group, key=lambda n: hashes[n][1][0] * hashes[n][1][1], reverse=True)
        keep = reps[0]
        keep_near.append(keep)
        print(f"\nGroup {gi} ({len(group)} images) — keep: {keep}")
        for n in sorted(group):
            if n == keep:
                print(f"  [KEEP]   {n}")
            else:
                remove_near.append(n)
                print(f"  [REMOVE] {n}")

    # Loosely similar (same scene, different framing)
    used_loose = set(used)
    loose_groups: list[list[str]] = []
    for i, n1 in enumerate(names):
        if n1 in used_loose:
            continue
        group = [n1]
        ph1 = hashes[n1][0]
        for n2 in names[i + 1 :]:
            if n2 in used_loose:
                continue
            d = ph1 - hashes[n2][0]
            if TH_NEAR < d <= TH_LOOSE:
                group.append(n2)
        if len(group) > 1:
            used_loose.update(group)
            loose_groups.append(group)

    remove_loose: list[str] = []
    print(f"\n=== SIMILAR SCENES (pHash {TH_NEAR + 1}-{TH_LOOSE}, optional trim): {len(loose_groups)} groups ===")
    for gi, group in enumerate(loose_groups, 1):
        reps = sorted(group, key=lambda n: hashes[n][1][0] * hashes[n][1][1], reverse=True)
        keep = reps[0]
        print(f"\nLoose group {gi} ({len(group)} images) — keep: {keep}")
        for n in sorted(group):
            if n == keep:
                print(f"  [KEEP]   {n}")
            else:
                remove_loose.append(n)
                print(f"  [REMOVE?] {n}")

    exact_remove = sum(len(g) - 1 for g in exact)
    print("\n=== SUMMARY ===")
    print(f"Total:                    {len(files)}")
    print(f"Exact dupes to remove:    {exact_remove}")
    print(f"Near-dupes to remove:     {len(remove_near)}")
    print(f"Loose similar (optional): {len(remove_loose)}")
    print(f"After strict dedup:       ~{len(files) - exact_remove - len(remove_near)}")
    print(f"After full dedup:         ~{len(files) - exact_remove - len(remove_near) - len(remove_loose)}")


def analyze_car_session():
    car_files = sorted(
        f
        for f in REAL.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and " at 20.3" in f.name
        and " at 20.39" not in f.name
    )
    hashes = {f.name: imagehash.phash(Image.open(f)) for f in car_files}
    names = list(hashes.keys())
    th = 12

    clusters: list[list[str]] = []
    used: set[str] = set()
    for n1 in names:
        if n1 in used:
            continue
        cluster = [n1]
        for n2 in names:
            if n2 == n1 or n2 in used:
                continue
            if hashes[n1] - hashes[n2] <= th:
                cluster.append(n2)
        used.update(cluster)
        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    print(f"\n=== CAR/STREET SESSION ({len(names)} photos, pHash <= {th}) ===")
    print(f"Scene clusters: {len(clusters)} (keep ~1 per cluster for max diversity)\n")

    remove: list[str] = []
    for i, cluster in enumerate(clusters, 1):
        keep = max(cluster, key=lambda n: Image.open(REAL / n).size[0] * Image.open(REAL / n).size[1])
        print(f"Cluster {i} ({len(cluster)} imgs) — KEEP: {keep}")
        for n in sorted(cluster):
            if n != keep:
                remove.append(n)
                print(f"  [REMOVE] {n}")

    print(f"\nCar session: keep {len(clusters)}, remove {len(remove)}")


if __name__ == "__main__":
    main()
    analyze_car_session()
