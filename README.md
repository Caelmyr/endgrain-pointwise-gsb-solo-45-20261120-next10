# Endgrain Lab

Endgrain Lab is a local workbench for inspecting wood end-face photographs. It keeps each original image with an editable measuring setup, revision-bound analyses, observations, and issued reports.

## Start

```sh
docker compose up -d --build --wait
```

Open http://127.0.0.1:18948. Use `ENDGRAIN_PORT` to select another loopback port. Compose stores originals, measuring state, and reports in the `inspections` volume. `docker compose stop` and `docker compose down` leave the volume available for a later start.

For a Python 3.11 environment, install `requirements.txt` and run `./start.sh`. The default local data directory is `.endgrain/`. Set `ENDGRAIN_DATA` to use another directory. The Docker build can take an `ENDGRAIN_DEPS_IMAGE` build argument for a cached Python 3.11 image containing the pinned dependencies. Use the same architecture and Python ABI as the runtime base.

## Typical inspection

1. Import a single face photograph or open the synthetic samples. The synthetic white oak has a branching check and growth-ring bands. The synthetic walnut includes a brighter area and two tones. The synthetic ash has periodic saw marks.
2. Calibrate with a measured ruler or rectify four corners of a known rectangle. Click the source points in order. The Grid view displays the selected plane and the ruler length or grid spacing. Draw a simple polygon using **Region** to limit analysis.
3. Optionally attach a flat-field reference from the same setup. The reference must have the same oriented dimensions. Use the Lighting view to check gain and unsupported reference locations.
4. Run a measurement. Results include an overlay, summary, and a downloadable CSV. Use the Pixel probe on the working image for a magnified patch, source mapping and luminance profiles. Add review markers by clicking locations in the selected measuring region.
5. Plan the micrograph round under **Micrograph stations**. The plan is saved with the inspection and reopens with it. Calibrate first, outline two representative faces (A and B) with the face buttons, then set the field size in millimetres, the minimum number of frames each face needs, the frame budget and the minimum gap between fields. Stations are placed only where the whole field sits on photographed, clear pixels, and no two frames overlap. A face that cannot reach its minimum is reported with the missing count and the dominant obstacle; spare frames on the other face are labelled as coverage only, never as a stand-in for the short face. Mark a station **Void** on site to keep its position as a crossed-out record and receive a replacement candidate on the same face; captured frames never move. **Open station plan** shows the numbered overlay, and the walk list CSV carries the walking order, millimetre centre and top-left corner, and the source pixel of each centre.
6. Compare two saved measurements of the same kind. A color difference overlay is available when they share a source, plane, region and lighting settings. The Recipes tab can replay selected tool settings on a second inspection. The Lots tab screens saved readings and exports a slide of face overlays and a CSV.
7. Include current measurements in a report and download its evidence ZIP. It contains source material, the measuring setup illustration, overlays, tables, the manifest and checksums. The separate inspection exchange ZIP carries the original capture, settings, attached lighting reference and current markers into a separate record on restore.

## Image measurements

- **Capture quality** shows channel clipping, low-detail regions, glare-like pixels and supported coverage.
- **Annual ring transect** samples a narrow band and suggests contrast boundaries. Inspect or replace the candidates before interpreting the distances as growth increments.
- **Crack traces** estimates skeleton lengths and dark-region widths. It identifies branch and edge-touching regions as review candidates.
- **Vessel pores** estimates component diameter, area and density from compact dark regions.
- **Grain orientation** maps axial texture and represents ambiguous local orientation separately from reliable regions.
- **Saw chatter** measures texture power within an axis and wavelength band in overlapping windows.
- **Wood footprint** isolates a warm, bright connected end-face silhouette against its backing.
- **Edge recession** shows supported loss or unknown areas along a manually specified nominal stock outline.
- **Color zones** compares appearance to swatches in display-referred Lab using the current correction settings.

## Acquisition limits

Inputs are PNG, JPEG, TIFF and WebP, one frame per import, up to 32 MiB and 16 megapixels with each side at least 48 pixels. The measuring raster has a maximum side of 1,800 pixels. Reported pixel widths use that raster. Fully opaque source pixels contribute to geometric support. EXIF orientation is applied before choosing coordinates and high bit-depth grayscale inputs are rendered to an 8-bit measuring representation. The unmodified input remains downloadable.

The rectangle method assumes a planar specimen and a known rectangular reference in that plane. Neither scale method corrects lens distortion. Inspection tools are approximate image measurements for operator review. The demonstration stock is procedurally generated and cannot establish botanical identification, defect acceptance limits or a calibrated color standard.

Micrograph plans belong to the revision that made them: after a calibration or rectification change, reset and re-plan instead of walking stale millimetre coordinates. Voided frames still count against the effort spent but not as coverage, so the shortfall stays visible until a replacement is actually captured.

A change to measuring setup starts a new revision. Older results stay in measurement history, and issued reports remain bound to the original revision and settings. A measurement recipe transfers settings, not the identity of a manual ring or defect.

## Source layout

`endgrain/sampling.py` holds the station planner, the clear-field screening and the marked plan image and walk list. `endgrain/` contains image ingest, geometry, measurement, SQLite, and report logic. `static/` contains the browser workbench. `app.py` runs the Waitress server. `scripts/count_backend.py` can count Python backend source.
