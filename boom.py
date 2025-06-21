import pathlib, tarfile, subprocess, sys
import build

# build the sdist only (in dist/)
builder = build.ProjectBuilder(".")
sdist = builder.build("sdist", "dist")

print("built sdist:", sdist)
print("--- content ---")
with tarfile.open(sdist) as tf:
    members = [m.name for m in tf.getmembers()]
    missing = [m for m in
               ("volkit/estimators", "volkit/estimators/__init__.py")
               if all(not s.startswith(m) for s in members)]
    print("\n".join(members))
    if missing:
        print("\nMISSING:", missing)