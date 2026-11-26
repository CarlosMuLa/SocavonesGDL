# SocavonesGDL

Geospatial and sparse-sample-safe statistical pipeline evaluating whether colonias in the **Guadalajara Metropolitan Area (ZMG / AMG)** with recurring/high flood exposure accumulate significantly more sinkholes than colonias with low flood exposure (**2020–2024** primary window; **2020–Oct 2026** extended dataset).

---

## Architecture Overview

| Stage | CLI Script | Core Modules | Primary Outputs |
|---|---|---|---|
| **1. Canonical 20 m Raster Stack** | `scripts/01_build_raster_stack.py` | `grid.py`, `stac_sentinel.py`, `landcover.py`, `terrain.py` | Pixel-aligned `EPSG:32613` 20 m GeoTIFFs (Sentinel-2 dry/wet/5yr bands + NDVI/NDWI/NDBI, ESA WorldCover fractional built-up + NDBI bare-soil check, Copernicus GLO-30 macro topography) |
| **2. Colonia Aggregation** | `scripts/02_aggregate_colonias.py` | `vector_prep.py`, `aggregate.py` | `colonias_analysis.parquet`, `colonias_analysis.csv`, `sinkhole_colonia_match_audit.csv`, `sinkhole_source_isolation_audit.json` |
| **3. Sparse-Sample Statistics** | `scripts/03_run_statistics.py` | `stats_models.py` | `results_summary.md`, `group_comparisons.csv`, `glm_coefficients.csv`, `spatial_cv_metrics.csv`, `statistical_results.json` |
| **4. AWS IaC & CI/CD** | `.github/workflows/ci.yml`, `.github/workflows/deploy.yml` | `infra/cloudformation.yaml`, `scripts/setup_ec2_env.sh` | Encrypted S3 bucket, SSM-enabled EC2 instance profile, optional AL2023 EC2 host, automatic push-to-`main` deployment |

### Key Methodological Guardrails
- **Zero hardcoded tiles:** Element 84 STAC (`sentinel-2-l2a`, `cop-dem-glo-30`) and ESA WorldCover (`esa_worldcover_grid.geojson`) are queried dynamically from the canonical `EPSG:32613` 20 m grid's WGS84 bounding box.
- **Direct S3 COG streaming:** Uses `AWS_NO_SIGN_REQUEST="YES"` and `GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"` — never downloads full raw Sentinel-2 archives.
- **Uniform 20 m & Baseline 04.00+ consistency:** All Sentinel-2 bands (`red`, `green`, `nir`, `swir16`) are harmonized to 20 m (`average` for 10 m bands, `bilinear` for `swir16`), with automatic STAC `raster:bands` scale/offset handling across 2020–2021 vs. 2022+.
- **NDBI cross-check:** Built-up land cover uses ESA WorldCover 10 m class-50 aggregated to fractional 20 m built-up (`0–1`), with an explicit dry bare-soil confusion audit against NDBI (plus optional local Dynamic World GeoTIFF hook).
- **Macro topography caveat:** Copernicus GLO-30 derivatives (`elevation`, `slope`, `TPI`, `depression_depth`, `coarse TWI`) are explicitly tagged as coarse terrain proxies that do not resolve curbs, buildings, or underground storm drains.
- **Strict source isolation:** Every sinkhole record must carry `source` in `{press_verified, official, crowdsourced}`; `crowdsourced` records are strictly excluded from every primary statistical test and regression model.
- **Sparse-sample statistical hierarchy:** Nonparametric Mann-Whitney U + 10,000-resample permutation tests + Fisher's exact test → parsimonious Poisson & Negative Binomial GLMs with `log(area_km2)` offset & Logistic regression (3–5 predictors, VIF audited) → leakage-free Leave-One-Municipality-Out (LOMO) and 5 km Spatial Block CV. Optional shallow ML is quarantined behind `--include-exploratory-ml`.

---

## Automated CI/CD & AWS Resource Deployment

### Workflows Included
1. **`.github/workflows/ci.yml`** (runs on every push & PR):
   - Runs Python 3.9 `pytest` suite (`tests/`), validates `config/zmg_config.yaml`, checks `scripts/setup_ec2_env.sh` with `bash -n`, and lints `infra/cloudformation.yaml` with `cfn-lint`.
2. **`.github/workflows/deploy.yml`** (runs on push to `main` and manual `workflow_dispatch`):
   - Runs full Python 3.9 tests + `cfn-lint` gate.
   - Deploys `infra/cloudformation.yaml` in **`us-west-2`** (encrypted versioned S3 data/outputs bucket with `Retain` policy, least-privilege SSM-enabled EC2 IAM Role & Instance Profile, and optional AL2023 EC2 compute instance).
   - Syncs `config/` and `data/templates/` to S3.
   - Automatically updates `/home/ec2-user/SocavonesGDL` and `~/geo_env` on your EC2 instance via **AWS SSM Run Command** (recommended keyless path) or **SSH fallback**.

### One-Time GitHub & AWS Setup
1. **(Recommended, keyless OIDC) Bootstrap the CloudFormation stack once from AWS CLI** (creates S3 bucket, EC2 instance profile, and GitHub OIDC deploy role):
   ```bash
   aws cloudformation deploy \
     --region us-west-2 \
     --stack-name socavones-gdl-stack \
     --template-file infra/cloudformation.yaml \
     --capabilities CAPABILITY_IAM \
     --parameter-overrides \
       GitHubRepo=CarlosMuLa/SocavonesGDL \
       GitHubBranch=main \
       CreateGitHubOidcRole=true \
       CreateGitHubOidcProvider=true \
       ProvisionEc2Instance=false
   ```
   *(If your AWS account already has a GitHub OIDC provider, set `CreateGitHubOidcProvider=false`.)*
2. **Read stack outputs** to get `GitHubDeployRoleArn`, `DataBucketName`, and `Ec2InstanceProfileName`:
   ```bash
   aws cloudformation describe-stacks \
     --region us-west-2 \
     --stack-name socavones-gdl-stack \
     --query "Stacks[0].Outputs" \
     --output table
   ```
3. **Attach `Ec2InstanceProfileName`** to your existing Amazon Linux 2023 EC2 instance in `us-west-2` (or set `ProvisionEc2Instance=true` via manual workflow dispatch to launch a new SSM-managed AL2023 instance).
4. **Configure GitHub Repository Secrets/Variables** (*Settings → Secrets and variables → Actions*):
   - **Secret `AWS_ROLE_TO_ASSUME`**: `GitHubDeployRoleArn` from step 2 *(or fallback `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`)*.
   - **Variable or Secret `EC2_INSTANCE_ID`**: Your `us-west-2` EC2 instance ID (e.g. `i-0abc12345def67890`) for keyless SSM auto-deploy.
   - *(Optional SSH fallback if not using SSM)*: Secrets `EC2_HOST` and `EC2_SSH_KEY` (plus optional variable `EC2_USER`, defaults to `ec2-user`).

---

## Running on AWS EC2 (Amazon Linux 2023, `~/geo_env`)

### 1. Environment Bootstrap (`dnf`-only, preserves preinstalled `rasterio`/`numpy`/`pystac-client`)
```bash
bash scripts/setup_ec2_env.sh
source ~/geo_env/bin/activate
```

### 2. Stage 1 — Build the 20 m Canonical ZMG Raster Stack (Runs Immediately)
Stage 1 uses the 9-municipality ZMG fallback bbox automatically if colonia polygons are not yet in `data/raw/`, or derives & snaps the grid from `--colonias` once available:
```bash
# Quick metadata & tile-discovery check (no pixel download)
python scripts/01_build_raster_stack.py --dry-run

# Full 20 m raster stack (Sentinel-2 + ESA WorldCover + Copernicus GLO-30)
python scripts/01_build_raster_stack.py --step all --output-dir outputs/rasters

# Or run individual stages independently
python scripts/01_build_raster_stack.py --step sentinel --output-dir outputs/rasters
python scripts/01_build_raster_stack.py --step worldcover --output-dir outputs/rasters
python scripts/01_build_raster_stack.py --step dem --output-dir outputs/rasters
```

### 3. Prepare Raw Vector & Tabular Datasets (`data/raw/`)
Place the following direct public / curated datasets in `data/raw/` (CSV schema templates are provided in `data/templates/`):
- **`data/raw/colonias_zmg.gpkg`** (or `.shp`): Download directly from **IIEG Jalisco** (`https://iieg.gob.mx/ns/?page_id=882`, *"Shape Información Sociodemográfica por colonia, Jalisco (Colonias enteras)"*) or **INEGI DCAH** (Jalisco).
- **`data/raw/flood_exposure.geojson`** (or `.shp` / `.csv`): Static recurring-flood sites from **Imeplan MUI / Zoom Metropolitano** (`https://mui.imeplan.mx`, `https://zoom.imeplan.mx/`) and/or **González, Comi & López De la Cruz (2025)** (*Tecnología y Ciencias del Agua*, DOI `10.24850/j-tyca-2025-02-06` — note Guadalajara-only scope guardrail).
- **`data/raw/sinkholes_zmg.csv`**: Audited, deduplicated `press_verified` sinkhole ledger matching `data/templates/sinkholes_template.csv` (required because official agencies confirmed no georeferenced sinkhole location registry exists; tag any crowdsourced entries `source=crowdsourced` so they stay strictly isolated).

### 4. Stage 2 — Colonia-Level Aggregation
```bash
python scripts/02_aggregate_colonias.py \
  --raster-dir outputs/rasters \
  --colonias data/raw/colonias_zmg.gpkg \
  --flood-path data/raw/flood_exposure.geojson \
  --flood-type points \
  --sinkholes data/raw/sinkholes_zmg.csv \
  --output-dir outputs/tables
```

### 5. Stage 3 — Sparse-Sample Statistical Analysis & Spatial CV
```bash
# Primary analysis (nonparametric + Poisson/NB & Logistic GLMs + LOMO & Spatial Block CV)
python scripts/03_run_statistics.py \
  --input outputs/tables/colonias_analysis.parquet \
  --output-dir outputs/stats

# Optional: include secondary exploratory ML benchmarks on identical spatial CV folds
python scripts/03_run_statistics.py \
  --input outputs/tables/colonias_analysis.parquet \
  --output-dir outputs/stats \
  --include-exploratory-ml
```
