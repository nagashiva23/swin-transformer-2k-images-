"""Interactive local demo for the ROCOv2 Swin and CNN+LSTM captioners."""

from __future__ import annotations

import io
import csv
import json
import queue
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parent
CNN_DIR = ROOT / "CNN+LSTM"
sys.path.insert(0, str(CNN_DIR))

from compare_models_results import DEVICE, decode, gpu_info, load_model  # noqa: E402

DATA_ROOT = Path(r"C:\AIE Files\Projects\S5\Dl\DATA\rocov2")
SWIN_CHECKPOINT = DATA_ROOT / "swin_caption_best.pt"
CNN_CHECKPOINT = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "best.pt"
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.set_page_config(page_title="Swin Transformer + CNN LSTM", layout="wide")


@st.cache_resource(show_spinner="Loading model checkpoint…")
def cached_model(kind: str, checkpoint: str):
    return load_model(kind, Path(checkpoint))


def to_tensor(image: Image.Image) -> torch.Tensor:
    return TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def prediction_worker(kind: str, model, vocabulary, max_length: int, tensor: torch.Tensor, events):
    """Run encoder and greedy decoder explicitly so the UI can show real stages."""
    events.put((kind, "encoder_start", None))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    started = time.perf_counter()
    stage_times = {}
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    hooks = []
    active_layer_timers = {}

    def add_stage_hooks(module, start_map, done_map):
        for module_index, submodule in enumerate(module):
            start_stage = start_map.get(module_index)
            done_stage = done_map.get(module_index)
            if start_stage is not None:
                def on_start(_module, _inputs, stage=start_stage):
                    active_layer_timers[stage] = time.perf_counter()
                    events.put((kind, "layer_start", stage))
                hooks.append(submodule.register_forward_pre_hook(on_start))
            if done_stage is not None:
                def on_done(_module, _inputs, _output, stage=done_stage):
                    seconds = time.perf_counter() - active_layer_timers.pop(stage, time.perf_counter())
                    events.put((kind, "layer_done", (stage, seconds)))
                hooks.append(submodule.register_forward_hook(on_done))

    if kind == "swin":
        feature_children = list(model.encoder.features.children())
        swin_starts = {0: 1, 1: 2, 2: 3, 4: 4, 6: 5}
        swin_ends = {0: 1, 1: 2, 3: 3, 5: 4, 7: 5}
        add_stage_hooks(feature_children, swin_starts, swin_ends)
        norm_started = {}
        def norm_start(_module, _inputs):
            norm_started["time"] = time.perf_counter()
            events.put((kind, "layer_start", 6))
        def norm_done(_module, _inputs, _output):
            events.put((kind, "layer_done", (6, time.perf_counter() - norm_started.get("time", time.perf_counter()))))
        hooks.append(model.encoder.norm.register_forward_pre_hook(norm_start))
        hooks.append(model.encoder.norm.register_forward_hook(norm_done))
    else:
        backbone_children = list(model.encoder.backbone.children())[:-1]
        cnn_starts = {0: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        cnn_ends = {3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
        add_stage_hooks(backbone_children, cnn_starts, cnn_ends)

    with torch.inference_mode():
        with (torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()):
            if kind == "swin":
                memory = model.encoder(tensor)
                feature_map = memory.float().norm(dim=-1).reshape(7, 7)
            else:
                spatial = model.encoder.backbone[:-1](tensor)
                feature_map = spatial.float().norm(dim=1).squeeze(0)
                pool_started = time.perf_counter()
                events.put((kind, "layer_start", 6))
                pooled = torch.nn.functional.adaptive_avg_pool2d(spatial, 1).flatten(1)
                image_features = model.encoder.projection(pooled)
                events.put((kind, "layer_done", (6, time.perf_counter() - pool_started)))
        for hook in hooks:
            hook.remove()
        stage_times["encoder_seconds"] = time.perf_counter() - started
        events.put((kind, "encoder_done", stage_times["encoder_seconds"]))
        decoder_started = time.perf_counter()

        if kind == "swin":
            events.put((kind, "decoder_start", 7))
            ids = torch.full((tensor.size(0), 1), vocabulary.sos_id, dtype=torch.long, device=DEVICE)
            finished = torch.zeros(tensor.size(0), dtype=torch.bool, device=DEVICE)
            for step in range(max_length - 1):
                events.put((kind, "decoder_compute_start", step + 1))
                with (torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()):
                    logits = model.decoder(ids, memory)
                step_logits = logits[:, -1, :].clone()
                previous = ids[:, -1]
                for batch_index in range(tensor.size(0)):
                    if not finished[batch_index]:
                        token_id = previous[batch_index].item()
                        if token_id not in (vocabulary.eos_id, vocabulary.pad_id):
                            step_logits[batch_index, token_id] = -100.0
                        if ids.size(1) >= 2:
                            sequence = ids[batch_index].tolist()
                            seen = {(sequence[i], sequence[i + 1], sequence[i + 2])
                                    for i in range(len(sequence) - 2)}
                            prefix = (sequence[-2], sequence[-1])
                            banned = {trigram[2] for trigram in seen if trigram[:2] == prefix}
                            if banned:
                                step_logits[batch_index, list(banned)] = -100.0
                next_ids = step_logits.argmax(-1, keepdim=True)
                next_ids = torch.where(finished.unsqueeze(1), torch.full_like(next_ids, vocabulary.pad_id), next_ids)
                ids = torch.cat([ids, next_ids], dim=1)
                finished = finished | (next_ids.squeeze(1) == vocabulary.eos_id)
                events.put((kind, "decoder_progress", (step + 1, time.perf_counter() - decoder_started)))
                if finished.all():
                    break
        else:
            decoder = model.decoder
            events.put((kind, "lstm_init_start", None))
            with (torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()):
                hidden, cell = decoder._initial_state(image_features)
            events.put((kind, "decoder_start", 8))
            current = torch.full((tensor.size(0), 1), vocabulary.bos_id, dtype=torch.long, device=DEVICE)
            generated = [current]
            for step in range(max_length - 1):
                with (torch.autocast("cuda", dtype=torch.float16) if DEVICE.type == "cuda" else nullcontext()):
                    embeddings = decoder.input_dropout(decoder.embedding(current[:, -1:]))
                    output, (hidden, cell) = decoder.lstm(embeddings, (hidden, cell))
                    logits = decoder.output(decoder.output_dropout(output[:, -1, :]))
                next_ids = logits.argmax(dim=-1, keepdim=True)
                generated.append(next_ids)
                current = next_ids
                events.put((kind, "decoder_progress", (step + 1, time.perf_counter() - decoder_started)))
                if (next_ids == vocabulary.eos_id).all():
                    break
            ids = torch.cat(generated, dim=1)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    stage_times["decoder_seconds"] = time.perf_counter() - decoder_started
    elapsed = time.perf_counter() - started
    caption = decode(vocabulary, ids[0].detach().cpu())
    events.put((kind, "done", None))
    return {"kind": kind, "caption": caption, "seconds": elapsed,
            "feature_map": feature_map.detach().float().cpu().tolist(),
            "words": len(caption.split()), **stage_times}


@st.cache_data
def read_test_gallery(data_root: str, max_images: int = 20):
    root = Path(data_root)
    captions_path = root / "test_captions.csv"
    image_dir = root / "test_images" / "test"
    if not captions_path.is_file() or not image_dir.is_dir():
        return []
    gallery = []
    with captions_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            image_id = row.get("ID", "")
            path = next((image_dir / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
                         if (image_dir / f"{image_id}{suffix}").is_file()), None)
            if path:
                gallery.append({"id": image_id, "caption": row.get("Caption", ""), "path": str(path)})
            if len(gallery) >= max_images:
                break
    return gallery


def render_flow(kind: str, states: list[str], details: dict[int, str] | None = None) -> None:
    title = "Swin Transformer" if kind == "swin" else "CNN + LSTM"
    stages = (['Image + normalize', 'Patch embed', 'Swin stage 1', 'Merge + stage 2',
               'Merge + stage 3', 'Merge + stage 4', 'Final norm + 49 tokens',
               'Transformer decode', 'Token + EOS control', 'Caption text']
              if kind == "swin" else
              ['Image + normalize', 'ResNet conv stem', 'Residual layer 1', 'Residual layer 2',
               'Residual layer 3', 'Residual layer 4', 'Pool + 512d projection',
               'Initialize LSTM state', 'Token loop + EOS', 'Caption text'])
    nodes, edges = [], []
    details = details or {}
    for index, (label, state) in enumerate(zip(stages, states)):
        caption = details.get(index, state.capitalize())
        nodes.append(f'<div class="stage {state}"><span class="stage-index">{index + 1}</span>'
                     f'<strong>{label}</strong><small>{caption}</small></div>')
        if index < len(stages) - 1:
            edge_state = "flowing" if states[index] == "done" and states[index + 1] == "active" else ""
            edges.append(f'<div class="flow-edge {edge_state}"></div>')
    track = []
    for index, node in enumerate(nodes):
        track.append(node)
        if index < len(edges):
            track.append(edges[index])
    columns = " ".join(["minmax(0,1fr) .14fr"] * (len(stages) - 1) + ["minmax(0,1fr)"])
    html = f"""<style>
    .model-flow {{background:linear-gradient(135deg,#0b1224,#101b32);border:1px solid #263752;
      border-radius:18px;padding:17px 18px 20px;margin:6px 0 12px;color:#e5efff;}}
    .flow-title {{font:700 17px system-ui;margin:0 0 16px;color:#edf5ff;}}
    .flow-track {{display:grid;grid-template-columns:{columns};align-items:center;gap:5px;}}
    .stage {{min-height:92px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;
      border:1px solid #34435c;background:#111a2a;border-radius:13px;padding:9px 7px;transition:all .25s ease;}}
    .stage strong {{font:600 11px system-ui;margin-top:6px;line-height:1.2;}}
    .stage small {{font:10px system-ui;color:#91a2bd;margin-top:5px;}}
    .stage-index {{height:21px;width:21px;border-radius:50%;display:grid;place-items:center;font:700 11px system-ui;background:#28364b;color:#b8c8df;}}
    .stage.done {{border-color:#22d3a6;background:#102b2b;box-shadow:0 0 13px #0dd5a633;}}
    .stage.done .stage-index {{background:#0b8b6b;color:#fff;}}
    .stage.active {{border-color:#46d9ff;background:#10243a;box-shadow:0 0 17px #29c9ff88;animation:pulse 1.3s ease-in-out infinite alternate;}}
    .stage.active .stage-index {{background:#0788b7;color:white;}}
    .flow-edge {{height:3px;background:#34435c;border-radius:3px;position:relative;overflow:hidden;}}
    .flow-edge.flowing {{background:#155064;box-shadow:0 0 9px #20caff66;}}
    .flow-edge.flowing:after {{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,#56edff,transparent);
      background-size:200% 100%;animation:flow 1s linear infinite;box-shadow:0 0 10px #56edff;}}
    @keyframes flow {{from{{background-position:100% 0}}to{{background-position:-100% 0}}}}
    @keyframes pulse {{from{{transform:scale(1);box-shadow:0 0 7px #29c9ff55}}to{{transform:scale(1.025);box-shadow:0 0 20px #29c9ffaa}}}}
    @media(max-width:1050px){{.flow-track{{grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}}.flow-edge{{display:none}}.stage{{min-height:76px}}}}
    @media(max-width:700px){{.flow-track{{grid-template-columns:repeat(2,minmax(0,1fr))}}.stage{{min-height:70px}}}}
    </style><section class="model-flow"><div class="flow-title">{title}</div>
    <div class="flow-track">{''.join(track)}</div></section>"""
    # Use Streamlit's HTML renderer so indentation in the generated markup is
    # not interpreted as a Markdown code block.
    st.html(html)


def read_training_history(kind: str):
    if kind == "swin":
        path = ROOT / "training_history.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))
    path = CNN_DIR / "artifacts" / "cnn_lstm_roco_full" / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def saved_metrics(kind: str):
    directory = ROOT / "result" / "swin" if kind == "swin" else CNN_DIR / "result"
    path = directory / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> None:
    hardware = gpu_info()
    st.title("Swin Transformer + CNN LSTM")
    st.caption("Interactive deep-learning demo: compare two image-captioning models and follow each model's processing stages.")
    info_cols = st.columns(4)
    info_cols[0].metric("Compute device", str(DEVICE).upper())
    info_cols[1].metric("GPU", hardware["gpu_name"])
    info_cols[2].metric("VRAM", f"{hardware['gpu_memory_total_gib']:.1f} GiB" if hardware["gpu_memory_total_gib"] else "N/A")
    info_cols[3].metric("CUDA", str(hardware["cuda_runtime"] or "N/A"))
    st.info("This project generates captions. It does not contain an image-classification head or predict a diagnostic class.")

    tab_predict, tab_benchmarks, tab_training = st.container(), st.container(), st.container()
    with tab_predict:
        gallery = read_test_gallery(str(DATA_ROOT), max_images=20)
        source = st.radio("Image source", ["Built-in ROCO test gallery", "Upload another image"], horizontal=True)
        image, reference_caption, image_label = None, None, ""
        if source == "Built-in ROCO test gallery":
            if not gallery:
                st.warning(f"No ROCO test images found under {DATA_ROOT}. Use the upload option or update DATA_ROOT in app.py.")
            else:
                gallery_map = {row["id"]: row for row in gallery}
                image_ids = list(gallery_map)
                if st.session_state.get("gallery_image_id") not in gallery_map:
                    st.session_state.gallery_image_id = image_ids[0]
                shuffle_col, picker_col = st.columns([1, 3])
                with shuffle_col:
                    if st.button("Shuffle test image", use_container_width=True):
                        options = [value for value in image_ids if value != st.session_state.gallery_image_id]
                        st.session_state.gallery_image_id = random.choice(options or image_ids)
                with picker_col:
                    chosen_id = st.selectbox("Choose from 20 ROCO test images", image_ids,
                                             index=image_ids.index(st.session_state.gallery_image_id),
                                             format_func=lambda value: f"{value} | {gallery_map[value]['caption'][:72]}",
                                             key="gallery_image_id")
                chosen = gallery_map[chosen_id]
                image = Image.open(chosen["path"]).convert("RGB")
                reference_caption, image_label = chosen["caption"], chosen["id"]
                with st.expander("Browse the built-in image thumbnails"):
                    thumb_cols = st.columns(5)
                    for index, row in enumerate(gallery):
                        with thumb_cols[index % 5]:
                            st.image(row["path"], caption=row["id"], use_container_width=True)
        else:
            uploaded = st.file_uploader("Drag an image here or browse", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                        accept_multiple_files=False,
                                        help="Images are processed locally and are not saved by this app.")
            if uploaded is not None:
                try:
                    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                    image_label = uploaded.name
                except (OSError, ValueError) as error:
                    st.error(f"Could not read this image: {error}")

        mode = st.radio("Prediction mode", ["Compare both models in parallel", "Swin Transformer only", "CNN-LSTM only"], horizontal=True)
        selected = ["swin", "cnn_lstm"] if mode == "Compare both models in parallel" else (
            ["swin"] if mode == "Swin Transformer only" else ["cnn_lstm"])
        flow_states = {kind: (["done"] + ["idle"] * 9 if image is not None else ["idle"] * 10)
                       for kind in ("swin", "cnn_lstm")}
        flow_details = {kind: {} for kind in ("swin", "cnn_lstm")}
        flow_started = {kind: {} for kind in ("swin", "cnn_lstm")}
        st.markdown("#### Model pipeline")
        flow_slots = [st.empty(), st.empty()]

        def draw_flows():
            for slot, kind in zip(flow_slots, ("swin", "cnn_lstm")):
                with slot.container():
                    render_flow(kind, flow_states[kind], flow_details[kind])

        draw_flows()
        if image is None:
            st.info("Choose a built-in test image or drop in your own image to start the prediction flow.")
        else:
            image_col, result_col = st.columns([0.9, 1.1])
            with image_col:
                st.image(image, caption=image_label, use_container_width=True)
                if reference_caption:
                    st.markdown("**Dataset reference caption (test images only)**")
                    st.write(reference_caption)
                st.caption("A reference caption is available only for built-in test images. Uploaded images have no reference label.")
            with result_col:
                st.markdown("#### Predicted captions")
                run = st.button("Run prediction", type="primary", use_container_width=True)
                if run:
                    tensor = to_tensor(image)
                    loaded, missing = {}, []
                    for kind in selected:
                        checkpoint = SWIN_CHECKPOINT if kind == "swin" else CNN_CHECKPOINT
                        if not checkpoint.is_file():
                            missing.append(str(checkpoint))
                            continue
                        flow_states[kind] = ["active"] + ["idle"] * 9
                        flow_details[kind][0] = "Loading checkpoint..."
                        draw_flows()
                        try:
                            loaded[kind] = cached_model(kind, str(checkpoint))
                        except Exception as error:
                            st.error(f"Could not load {kind} model: {error}")
                    if missing:
                        st.error("Checkpoint not found:\n" + "\n".join(missing))
                    if len(loaded) == len(selected):
                        events = queue.Queue()
                        max_lengths = {kind: values[2] for kind, values in loaded.items()}

                        def apply_stage_event(kind, event, value):
                            if event == "encoder_start":
                                flow_states[kind] = ["done", "active"] + ["idle"] * 8
                                flow_started[kind][1] = time.perf_counter()
                                flow_details[kind][1] = "Extracting features..."
                            elif event == "layer_start":
                                stage = value
                                flow_states[kind] = ["done"] * stage + ["active"] + ["idle"] * (9 - stage)
                                flow_started[kind][stage] = time.perf_counter()
                                flow_details[kind][stage] = "Running..."
                            elif event == "layer_done":
                                stage, seconds = value
                                flow_details[kind][stage] = f"{seconds:.2f}s"
                                flow_states[kind] = ["done"] * (stage + 1) + ["idle"] * (9 - stage)
                                if stage < 6:
                                    flow_states[kind][stage + 1] = "active"
                                    flow_started[kind][stage + 1] = time.perf_counter()
                                    flow_details[kind][stage + 1] = "Next stage..."
                            elif event == "encoder_done":
                                flow_states[kind] = ["done"] * 7 + ["active"] + ["idle"] * 2
                                flow_started[kind][7] = time.perf_counter()
                                if kind == "swin":
                                    flow_details[kind][7] = "Preparing transformer decoder..."
                                else:
                                    flow_details[kind][7] = "Preparing LSTM state..."
                            elif event == "lstm_init_start":
                                flow_states[kind] = ["done"] * 7 + ["active"] + ["idle"] * 2
                                flow_started[kind][7] = time.perf_counter()
                                flow_details[kind][7] = "Building hidden/cell state..."
                            elif event == "decoder_start":
                                stage = value
                                flow_states[kind] = ["done"] * stage + ["active"] + ["idle"] * (9 - stage)
                                flow_started[kind][stage] = time.perf_counter()
                                flow_details[kind][stage] = "Decoder active..."
                            elif event == "decoder_compute_start":
                                flow_states[kind] = ["done"] * 7 + ["active", "idle", "idle"]
                                flow_details[kind][7] = f"Transformer pass {value}..."
                            elif event == "decoder_progress":
                                token, seconds = value
                                flow_states[kind] = ["done"] * 8 + ["active", "idle"]
                                flow_details[kind][8] = f"Token {token}/{max_lengths[kind] - 1} | {seconds:.2f}s"
                            elif event == "done":
                                flow_states[kind] = ["done"] * 9 + ["active"]
                                flow_details[kind][9] = "Caption ready"

                        with ThreadPoolExecutor(max_workers=len(loaded)) as executor:
                            futures = {}
                            for kind, (model, vocabulary, max_length, _specials, _payload) in loaded.items():
                                futures[kind] = executor.submit(prediction_worker, kind, model, vocabulary,
                                                               max_length, tensor, events)
                            while not all(future.done() for future in futures.values()):
                                try:
                                    while True:
                                        kind, event, value = events.get_nowait()
                                        apply_stage_event(kind, event, value)
                                except queue.Empty:
                                    pass
                                now = time.perf_counter()
                                for active_kind in selected:
                                    active_stage = next((index for index, state in enumerate(flow_states[active_kind])
                                                         if state == "active"), None)
                                    if active_stage is not None and active_stage in flow_started[active_kind]:
                                        elapsed = now - flow_started[active_kind][active_stage]
                                        flow_details[active_kind][active_stage] = f"{elapsed:.2f}s elapsed"
                                draw_flows()
                                time.sleep(0.12)
                            while True:
                                try:
                                    apply_stage_event(*events.get_nowait())
                                except queue.Empty:
                                    break
                            draw_flows()
                        results = []
                        for kind, future in futures.items():
                            try:
                                results.append(future.result())
                                flow_states[kind] = ["done"] * 10
                                flow_details[kind][6] = f"Encoder {results[-1]['encoder_seconds']:.2f}s"
                                flow_details[kind][8] = f"Decoder {results[-1]['decoder_seconds']:.2f}s"
                            except Exception as error:
                                flow_states[kind] = ["done"] * 8 + ["active", "idle"]
                                flow_details[kind][8] = "Prediction failed"
                                st.error(f"{kind} prediction failed: {error}")
                        draw_flows()
                        if results:
                            st.markdown("#### Caption and inference result")
                            output_cols = st.columns(len(results))
                            for column, result in zip(output_cols, results):
                                name = "Swin Transformer" if result["kind"] == "swin" else "CNN-LSTM"
                                with column:
                                    st.markdown(f"**{name} predicted caption**")
                                    st.success(result["caption"] or "(empty caption)")
                                    st.metric("Model inference", f"{result['seconds'] * 1000:.1f} ms")
                                    st.caption(f"{result['words']} caption words | device: {DEVICE}")
                                    with st.expander(f"{name} encoder feature map"):
                                        st.caption("7x7 feature magnitude; this is not a clinical saliency map.")
                                        figure = px.imshow(result["feature_map"], color_continuous_scale="Turbo",
                                                          labels={"x": "Grid x", "y": "Grid y", "color": "Feature magnitude"})
                                        figure.update_layout(height=330, margin=dict(l=8, r=8, t=15, b=8))
                                        st.plotly_chart(figure, use_container_width=True)
                            if len(results) > 1:
                                timing_data = []
                                for row in results:
                                    name = "Swin Transformer" if row["kind"] == "swin" else "CNN-LSTM"
                                    for stage, field in (("Encoder", "encoder_seconds"), ("Caption decoder", "decoder_seconds")):
                                        timing_data.append({"Model": name, "Stage": stage, "Seconds": row[field]})
                                timing_figure = px.bar(timing_data, x="Model", y="Seconds", color="Stage", barmode="group",
                                                       title="Per-model computation time by stage")
                                st.plotly_chart(timing_figure, use_container_width=True)
                                latency_data = [{"Model": "Swin Transformer" if row["kind"] == "swin" else "CNN-LSTM",
                                                 "Inference time (ms)": row["seconds"] * 1000,
                                                 "Caption words": row["words"]} for row in results]
                                latency_figure = px.bar(latency_data, x="Model", y="Inference time (ms)", color="Model",
                                                        hover_data=["Caption words"], title="Parallel per-image inference time")
                                st.plotly_chart(latency_figure, use_container_width=True)

    st.divider()
    st.header("Caption results and token confusion matrices")
    st.caption("Confusion matrices compare teacher-forced next-word predictions. These are token metrics, not image-diagnosis accuracy.")
    summary_columns = st.columns(2)
    for column, kind, name in zip(summary_columns, ("swin", "cnn_lstm"), ("Swin Transformer", "CNN + LSTM")):
        with column:
            st.subheader(name)
            metrics = saved_metrics(kind)
            result_dir = ROOT / "result" / "swin" if kind == "swin" else CNN_DIR / "result"
            matrix_path = result_dir / f"{kind}_token_confusion_matrix.png"
            if matrix_path.is_file():
                st.image(str(matrix_path), caption="Teacher-forced token confusion matrix", use_container_width=True)
            else:
                st.info("No saved token confusion matrix. Run the model comparison evaluation to generate it.")
            if metrics:
                first_metrics = st.columns(3)
                token_accuracy = metrics.get("token_accuracy_exact_lexical")
                first_metrics[0].metric("Token accuracy", f"{token_accuracy:.1%}" if token_accuracy is not None else "N/A")
                first_metrics[1].metric("BLEU-1", f"{metrics['bleu1']:.3f}" if metrics.get("bleu1") is not None else "N/A")
                first_metrics[2].metric("BLEU-4", f"{metrics['bleu4']:.3f}" if metrics.get("bleu4") is not None else "N/A")
                second_metrics = st.columns(3)
                second_metrics[0].metric("ROUGE-L", f"{metrics['rougeL']:.3f}" if metrics.get("rougeL") is not None else "N/A")
                second_metrics[1].metric("CIDEr-D", f"{metrics['ciderD']:.3f}" if metrics.get("ciderD") is not None else "N/A")
                second_metrics[2].metric("Checkpoint epoch", str(metrics.get("checkpoint_epoch", "N/A")))
            else:
                st.info("No saved evaluation metrics found for this model yet.")

    with tab_benchmarks:
        st.divider()
        st.header("Model evaluation graphs")
        st.subheader("Saved held-out evaluation")
        rows = []
        for kind in ("swin", "cnn_lstm"):
            metrics = saved_metrics(kind)
            if metrics:
                rows.append({"Model": metrics.get("model", kind), "BLEU-1": metrics.get("bleu1"),
                             "BLEU-4": metrics.get("bleu4"), "ROUGE-L": metrics.get("rougeL"),
                             "CIDEr-D": metrics.get("ciderD"), "ms/image": metrics.get("model_compute_ms_per_image"),
                             "images/s": metrics.get("images_per_second_including_load"),
                             "Peak GPU GiB": metrics.get("peak_gpu_memory_gib")})
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            quality_fields = [field for field in ("BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D")
                              if any(row.get(field) is not None for row in rows)]
            if quality_fields:
                quality = px.bar(rows, x="Model", y=quality_fields, barmode="group",
                                 title="Saved caption metrics (higher is better)")
                st.plotly_chart(quality, use_container_width=True)
            timing_fields = [field for field in ("ms/image", "images/s", "Peak GPU GiB")
                             if any(row.get(field) is not None for row in rows)]
            if timing_fields:
                timing_columns = st.columns(len(timing_fields))
                for column, field in zip(timing_columns, timing_fields):
                    perf = px.bar(rows, x="Model", y=field, color="Model", title=field)
                    column.plotly_chart(perf, use_container_width=True)
        else:
            st.info("No saved evaluation metrics found yet. Run compare_models_results.py to populate each model's result folder.")
        st.caption("Saved metrics are from the last evaluation run. This demo does not imply that a caption is a diagnosis.")

    with tab_training:
        st.divider()
        st.header("Training graphs")
        traces = []
        for kind, label, valid_key in (("swin", "Swin Transformer", "val_loss"),
                                       ("cnn_lstm", "CNN+LSTM", "valid_loss")):
            for row in read_training_history(kind):
                traces.append({"Model": label, "Epoch": row.get("epoch"),
                               "Train loss": row.get("train_loss"), "Validation loss": row.get(valid_key),
                               "Epoch seconds": row.get("epoch_seconds")})
        loss_rows = [row for row in traces if row["Epoch"] is not None]
        if loss_rows:
            loss_fig = go.Figure()
            for label in ("Swin Transformer", "CNN+LSTM"):
                model_rows = [row for row in loss_rows if row["Model"] == label]
                if model_rows:
                    loss_fig.add_trace(go.Scatter(x=[row["Epoch"] for row in model_rows],
                                                  y=[row["Train loss"] for row in model_rows],
                                                  mode="lines+markers", name=f"{label} train"))
                    loss_fig.add_trace(go.Scatter(x=[row["Epoch"] for row in model_rows],
                                                  y=[row["Validation loss"] for row in model_rows],
                                                  mode="lines+markers", name=f"{label} validation"))
            loss_fig.update_layout(title="Training and validation loss", xaxis_title="Epoch",
                                   yaxis_title="Loss", hovermode="x unified")
            st.plotly_chart(loss_fig, use_container_width=True)
            timed = [row for row in loss_rows if row["Epoch seconds"] is not None]
            if timed:
                duration_fig = px.line(timed, x="Epoch", y="Epoch seconds", color="Model", markers=True,
                                       title="Actual training duration per logged epoch")
                st.plotly_chart(duration_fig, use_container_width=True)
            else:
                st.info("Existing history files contain no epoch duration values. The updated trainers record times for future or resumed epochs.")
        else:
            st.info("No training history was found. Train a model to populate the loss chart.")


if __name__ == "__main__":
    main()
