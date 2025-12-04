# EMIT Training Pipeline Config Example

```
file_re: "EMIT_L2A_RFL_\d{3}_(\d{8}T\d{6})\.nc"
input_dir: "/data/nlahaye/remoteSensing/EMIT_RED_SEA/"
input_glob: "EMIT_L2A_RFL*nc"

train_timestamps: ["20250127T085614", "20250127T085626"]

out_dir: "/data/nlahaye/output/Learnergy/EMIT_RED_SEA_HABS/"

run_uid: "emit_red_sea_habs"

config_dir: "../../config"
```

