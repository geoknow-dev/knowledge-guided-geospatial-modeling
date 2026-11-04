# Reproducibility workflow

## 1. Prepare data

Acquire the input rasters listed in `data/README.md`. Document sources and licences, then preprocess all layers to the study-specific CRS, extent, resolution and NoData convention. Copy `configs/data_paths.example.json` to a local untracked file and update the paths.

## 2. Configure the model

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `MODEL_NAME`. To match a local OpenAI-compatible server, use its local endpoint and exact checkpoint name. Record the provider/server version, model revision, quantization, hardware and run date.

## 3. Run controlled conditions

For each task, run all six cumulative levels from L1-1 through L3-2 with temperature 0.1 and conversation history disabled. Preserve every returned workflow, generated script, model response metadata and execution log.

## 4. Execute and evaluate

Run generated code in an isolated environment against the same aligned raster inputs. Compare workflows with the expert reference, code with the expert implementation, and output cells with the expert raster. Use `evaluation/metrics.py` for WVR, WGD, WKHR, WQI, lexical/syntactic/data-flow code measures, R², nRMSE, KCPI and KLG.

The lightweight AST/data-flow implementation in this repository is a transparent approximation. If the paper used the official CodeBLEU implementation, pin its exact repository commit and language parser resources instead of silently substituting this implementation.

## 5. Rebuild figures

Run `python scripts/plot_results.py`. The script writes 330-dpi PNG files and vector SVG files to `figures/generated/`.

## 6. Archive

Create a tagged GitHub release, record checksums, and archive the exact release with Zenodo or another long-term repository. The manuscript should cite the archived DOI rather than only a mutable branch URL.

