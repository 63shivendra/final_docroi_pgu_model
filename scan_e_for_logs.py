import os, sys, glob

print("=" * 70)
print("  Forensic Audit: Searching E:\\shivendra pratap singh for Logs")
print("=" * 70)

target_dir = r"E:\shivendra pratap singh"

if not os.path.exists(r"E:\\"):
    print("STATUS: Drive E:\\ is NOT mounted or disconnected.")
    sys.exit(0)

if not os.path.exists(target_dir):
    print(f"STATUS: E:\\ is mounted, but directory '{target_dir}' does not exist.")
    print("Root contents of E:\\:")
    try:
        for it in os.listdir(r"E:\\")[:20]:
            print(f"  {it}")
    except Exception as e:
        print(f"  Error reading E:\\: {e}")
    sys.exit(0)

print(f"STATUS: Target directory found: {target_dir}")
print("Scanning recursively for all .log, .json, .csv, .txt, .md, .pth files...")

extensions = ["*.log", "*.json", "*.csv", "*.txt", "*.md"]
found_files = []

for root, dirs, files in os.walk(target_dir):
    for f in files:
        low = f.lower()
        if any(low.endswith(ext.replace("*", "")) for ext in extensions):
            full_p = os.path.join(root, f)
            sz = os.path.getsize(full_p)
            found_files.append((full_p, sz))

print(f"\nTotal potential log/result files found: {len(found_files)}")
print("-" * 70)

# Highlight files mentioning 30k, 18k, eval, test, benchmark, doctamper, scd
matches = []
for p, sz in found_files:
    fname = os.path.basename(p).lower()
    if any(k in fname or k in p.lower() for k in ["eval", "test", "benchmark", "scd", "30k", "18k", "doctamper", "result", "metric", "report"]):
        matches.append((p, sz))
        print(f"[RELEVANT] {p} ({sz} bytes)")

if not matches:
    print("No evaluation/benchmark log files found with 'eval', 'test', 'scd', '30k', etc. in filename.")

print("\nListing all discovered text/data files:")
for p, sz in found_files[:40]:
    print(f"  {p} ({sz} bytes)")
if len(found_files) > 40:
    print(f"  ... and {len(found_files) - 40} more files.")
