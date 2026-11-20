const viewport = document.querySelector("#image-stage");
const shell = document.querySelector("#image-shell");
const picture = document.querySelector("#face-image");
const canvas = document.querySelector("#draw-layer");
const pointerLabel = document.querySelector("#cursor-label");
const hint = document.querySelector("#stage-hint");
const context = canvas.getContext("2d");

export class WorkbenchViewer {
  constructor(onPoint, onPick) {
    this.onPoint = onPoint;
    this.onPick = onPick;
    this.info = {};
    this.zoom = 1;
    this.picked = null;
    picture.addEventListener("load", () => this.resize());
    picture.addEventListener("error", () => {
      hint.hidden = false;
      hint.textContent = "The image could not be loaded. Reopen the inspection to retry.";
    });
    new ResizeObserver(() => this.resize()).observe(viewport);
    canvas.addEventListener("click", event => {
      if (this.info.drawing) this.onPoint(this.point(event));
      else if (this.info.mode === "segments" && this.onPick) this.onPick(this.pick(this.point(event)));
    });
    canvas.addEventListener("pointermove", event => {
      const [x, y] = this.point(event);
      const plane = this.info.plane;
      const unit = plane?.unit;
      if (unit === "mm" && ["working", "result", "segments"].includes(this.info.mode)) {
        const sx = (x + 0.5) * plane.mm_per_x;
        const sy = (y + 0.5) * plane.mm_per_y;
        pointerLabel.textContent = sx.toFixed(1) + " / " + sy.toFixed(1) + " mm";
      } else {
        pointerLabel.textContent = Math.round(x) + " / " + Math.round(y) + " px";
      }
    });
    canvas.addEventListener("pointerleave", () => { pointerLabel.textContent = "—"; });
    document.querySelectorAll("[data-zoom]").forEach(button => {
      button.addEventListener("click", () => {
        const action = button.dataset.zoom;
        this.zoom = action === "fit" ? 1 : Math.max(0.64, Math.min(4, this.zoom * (action === "down" ? 0.8 : 1.25)));
        this.resize();
      });
    });
  }

  resetZoom() {
    this.zoom = 1;
    this.picked = null;
    viewport.scrollTo(0, 0);
  }

  show(info) {
    this.info = info;
    const requested = info.url;
    if (picture.getAttribute("src") !== requested) picture.setAttribute("src", requested);
    if (info.mode !== "segments") this.picked = null;
    hint.hidden = !info.drawing;
    hint.textContent = info.drawing?.hint ?? "";
    canvas.style.cursor = info.drawing ? "crosshair" : info.mode === "segments" ? "pointer" : "default";
    document.querySelector("#image-label").textContent = info.title;
    this.resize();
  }

  resize() {
    const w = picture.naturalWidth;
    const h = picture.naturalHeight;
    if (!w || !h || !viewport.clientWidth) return;
    const ratio = Math.min((viewport.clientWidth - 16) / w, (viewport.clientHeight - 14) / h) * this.zoom;
    const width = Math.max(1, Math.round(w * ratio));
    const height = Math.max(1, Math.round(h * ratio));
    shell.style.width = width + "px";
    shell.style.height = height + "px";
    viewport.classList.toggle("overzoom", width > viewport.clientWidth || height > viewport.clientHeight);
    const factor = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * factor);
    canvas.height = Math.round(height * factor);
    context.setTransform(canvas.width / w, 0, 0, canvas.height / h, 0, 0);
    document.querySelector("#zoom-level").textContent = this.zoom === 1 ? "Fit" : Math.round(this.zoom * 100) + "%";
    this.paint(1 / ratio);
  }

  select(segment) {
    this.picked = segment;
    const w = picture.naturalWidth;
    if (!w) return;
    const shown = shell.clientWidth / w;
    this.paint(shown ? 1 / shown : 1);
  }

  point(evt) {
    const rect = canvas.getBoundingClientRect();
    const w = picture.naturalWidth;
    const h = picture.naturalHeight;
    if (!rect.width || !w) return [0, 0];
    const x = Math.min(w - 1, Math.max(0, (evt.clientX - rect.left) * w / rect.width - 0.5));
    const y = Math.min(h - 1, Math.max(0, (evt.clientY - rect.top) * h / rect.height - 0.5));
    return [Number(x.toFixed(2)), Number(y.toFixed(2))];
  }

  pick([x, y]) {
    const segments = this.info.segments || [];
    let best = null;
    let bestDistance = Infinity;
    const reach = point => {
      const path = segment.path?.length ? segment.path : [segment.start, segment.end];
      for (let index = 1; index < path.length; index++) {
        const [ax, ay] = path[index - 1];
        const [bx, by] = path[index];
        const dx = bx - ax;
        const dy = by - ay;
        const span = dx * dx + dy * dy;
        const t = span ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / span)) : 0;
        distance = Math.min(distance, Math.hypot(ax + t * dx - x, ay + t * dy - y));
      }
    };
    let distance;
    segments.forEach(segment => {
      distance = Infinity;
      reach(segment);
      if (distance < bestDistance) {
        bestDistance = distance;
        best = segment;
      }
    });
    const tolerance = Math.max(4, (this.info.plane?.width || 600) / 120);
    if (!best || bestDistance > tolerance) return null;
    return best;
  }

  path(points, tone, width, {closed = false, dashed = false, dotted = false} = {}) {
    if (!points?.length) return;
    context.strokeStyle = tone;
    context.fillStyle = tone;
    context.lineWidth = width;
    context.setLineDash(dashed ? [width * 3, width * 2] : []);
    context.beginPath();
    points.forEach(([x, y], i) => {
      if (i === 0) context.moveTo(x + 0.5, y + 0.5);
      else context.lineTo(x + 0.5, y + 0.5);
    });
    if (closed) context.closePath();
    context.stroke();
    context.setLineDash([]);
    if (dotted) points.forEach(([x, y]) => {
      context.beginPath();
      context.arc(x + 0.5, y + 0.5, width * 2.4, 0, Math.PI * 2);
      context.fill();
    });
  }

  label(position, text, size, color = "#f7f0d6") {
    const [x, y] = position;
    context.font = "600 " + (size * 10) + "px system-ui";
    const textWidth = context.measureText(text).width + 10 * size;
    context.fillStyle = "#1b231eee";
    context.fillRect(x + 7 * size, y - 15 * size, textWidth, size * 15);
    context.fillStyle = color;
    context.fillText(text, x + 12 * size, y - 4 * size);
  }

  paint(size) {
    const {drawing, mode, plane, region, notes, guide, segments} = this.info;
    context.clearRect(0, 0, picture.naturalWidth, picture.naturalHeight);
    if (mode === "source" && !drawing) {
      const cal = plane?.calibration;
      if (cal?.mode === "ruler") this.path([cal.a, cal.b], "#ece8ba", size * 2, {dotted: true});
      if (cal?.mode === "quadrilateral") this.path(cal.corners, "#ece8ba", size * 2, {closed: true, dotted: true});
    }
    if (mode === "working") {
      if (!drawing || drawing.kind !== "region") this.path(region, "#dfd3a7", size * 1.15, {dashed: true, closed: true});
      if (!drawing) this.path(guide, "#e2ad6e", size * 1.6, {dotted: true});
    }
    if (mode === "segments" && this.picked) {
      const path = this.picked.path?.length ? this.picked.path : [this.picked.start, this.picked.end];
      this.path(path, "#ffe9b0", size * 3, {});
      this.path([this.picked.start], "#ffe9b0", size * 2.4, {dotted: true});
      this.path([this.picked.end], "#ffe9b0", size * 2.4, {dotted: true});
      const spot = path[Math.floor(path.length / 2)];
      this.label(spot, this.picked.label, size);
    }
    if (["working", "result"].includes(mode) && !drawing) {
      (notes || []).forEach(note => {
        this.path([note.point], "#e8b170", size * 1.5, {dotted: true});
        this.label(note.point, note.label, size);
      });
    }
    if (drawing) {
      this.path(drawing.points, "#ffd091", size * 2, {dotted: true});
      drawing.points.forEach((point, ix) => this.label(point, String(ix + 1), size));
    }
  }
}
