"""Gradio demo: fine-grained attention heatmap for a soundscape text query.

Given a lat/lon and a soundscape text description (or a LLaVA-generated
caption), this demo downloads the satellite tile at that location, runs it
through the Sat2Sound FDT codebook together with the text, and renders a
heatmap overlay on the image showing where in the scene each selected
word/phrase most strongly attends.

Required environment:

    SAT2SOUND_CKPT         path to the trained Sat2Sound checkpoint
    BINGMAP_API_KEY        Bing/Azure Maps key (or put it in .secrets/bingmap_api.txt)

Optional: a LLaVA caption is generated if you leave the text box empty or
check the "Use LLaVA" option. That path needs a GPU with ~16 GB VRAM.

Run: ``python demos/sat2sound_map.py``
"""

import io
import os
import urllib.request
from argparse import Namespace

import folium
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from folium import plugins
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from demos.demo_config import bingmap_api, log_dir, metadata_config, sat2sound_ckpt
from src.engine import prepare_flant5_text_embeds, sat2soundModel
from utilities.utils import sat_transform


matplotlib.use("Agg")

SIZE_SCALE = {1: 300, 3: 900, 5: 1500}
AUDIO_SOURCE_MAP = {"yfcc": 0, "iNat": 1, "aporee": 2, "freesound": 3}
CAPTION_SOURCE_MAP = {"meta": 0, "qwen": 1, "pengi": 2}

LLAVA_PROMPT = (
    "USER: <image>\nWhat types of sounds can we expect to hear from the "
    "location captured by this aerial view image? Describe in up to two "
    "sentences.\nASSISTANT:"
)


def load_api_key(file_path):
    env_val = os.environ.get("BINGMAP_API_KEY", "").strip()
    if env_val:
        return env_val
    try:
        with open(file_path, "r") as f:
            key = f.read().strip()
        if key:
            os.environ["BINGMAP_API_KEY"] = key
            return key
    except OSError:
        pass
    print(
        f"[warn] No Bing/Azure Maps API key found at {file_path} or in BINGMAP_API_KEY. "
        "Tile downloads will fail."
    )
    return ""


def load_model(ckpt_path, device):
    pretrained_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = pretrained_ckpt["hyper_parameters"]
    hparams.setdefault("jsd_weight", 0)
    hparams.setdefault("precision", "full")
    hparams["meta_droprate"] = 0.0
    hparams["mode"] = "evaluate"
    model = sat2soundModel(Namespace(**hparams)).to(device)
    model.load_state_dict(pretrained_ckpt["state_dict"], strict=False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    return Namespace(**hparams), model


def download_satellite_tile(lat, lon, api_key, out_file, zoom=18, size_px=1500):
    url = (
        "http://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/"
        f"{float(lat)},{float(lon)}/{zoom}?mapSize={size_px},{size_px}&key={api_key}"
    )
    try:
        urllib.request.urlretrieve(url, out_file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download satellite tile for ({lat}, {lon}): {exc}. "
            "Check your Bing Maps API key and network connection."
        ) from exc


def make_llava_caption(image_path, device):
    """Optional LLaVA caption path. Only imported if the user opts in."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    model = (
        LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.open(image_path)
    inputs = processor(images=image, text=LLAVA_PROMPT, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=77, do_sample=False)
    decoded = processor.decode(out[0], skip_special_tokens=True)
    parts = decoded.split("ASSISTANT: ")
    return parts[1] if len(parts) > 1 else decoded


def prepare_batch(image_path, text_prompt, hparams, tokenizer, device, zoom_level, lat, lon):
    sat_tr = sat_transform(is_train=False, input_size=224, sat_type="bingmap", zoom_level=zoom_level)
    image = Image.open(image_path)
    image_tensor = sat_tr(image)
    if hparams.precision == "half":
        image_tensor = image_tensor.half()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    text_input = tokenizer(
        text_prompt,
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_patch_embeds, text_boolean_mask = prepare_flant5_text_embeds(text_input, device)

    metadata = {
        "audio_source": None, "caption_source": None,
        "sat_zoom_level": [zoom_level],
        "latlong": None, "time": None, "month": None,
        "time_valid": None, "month_valid": None,
    }
    mt = getattr(hparams, "metadata_type", "none")
    if "latlong" in mt:
        lat_f, lon_f = float(lat), float(lon)
        metadata["latlong"] = torch.tensor([
            np.sin(np.pi * lat_f / 90), np.cos(np.pi * lat_f / 90),
            np.sin(np.pi * lon_f / 180), np.cos(np.pi * lon_f / 180),
        ]).float()
    if "asource" in mt:
        metadata["audio_source"] = torch.tensor(AUDIO_SOURCE_MAP[metadata_config.audio_source]).long()
    if "tsource" in mt:
        metadata["caption_source"] = torch.tensor(CAPTION_SOURCE_MAP[metadata_config.caption_source]).long()
    if "time" in mt:
        metadata["time"] = torch.tensor([
            np.sin(2 * np.pi * metadata_config.time / 23),
            np.cos(2 * np.pi * metadata_config.time / 23),
        ]).float()
        metadata["time_valid"] = torch.tensor(True).long()
    if "month" in mt:
        metadata["month"] = torch.tensor([
            np.sin(2 * np.pi * metadata_config.month / 12),
            np.cos(2 * np.pi * metadata_config.month / 12),
        ]).float()
        metadata["month_valid"] = torch.tensor(True).long()

    for k in metadata:
        if metadata[k] is not None:
            metadata[k] = metadata[k].unsqueeze(0).to(device)

    return {
        "sat": image_tensor,
        "sat_zoom_level": metadata["sat_zoom_level"],
        "text_input": {"patch_embeds": text_patch_embeds, "boolean_mask": text_boolean_mask},
        "latlong": metadata["latlong"],
        "audio_source": metadata["audio_source"],
        "caption_source": metadata["caption_source"],
        "month": metadata["month"],
        "month_valid": metadata["month_valid"],
        "time": metadata["time"],
        "time_valid": metadata["time_valid"],
    }, text_input.input_ids


def compute_heatmap(model, hparams, batch, sat_type, device):
    sat_token_embeddings = model.satmae_backbone(
        batch["sat"], zoom_level=batch["sat_zoom_level"], sat_type=sat_type
    )
    if hparams.metadata_type != "none":
        sat_token_embeddings = model.sat_encoder(
            sat_embeddings=sat_token_embeddings,
            latlong=batch["latlong"],
            audio_source=batch["audio_source"],
            caption_source=batch["caption_source"],
            time=batch["time"],
            month=batch["month"],
            time_valid=batch["time_valid"],
            month_valid=batch["month_valid"],
        )
    att_weight_img, _, _ = model.fdt.extract_img_sd_ft(sat_token_embeddings, return_token_att=True)

    text_patch_embeds, text_boolean_mask = model.text_encoder(batch["text_input"], embed_type="hidden_states")
    pad_mask = torch.where(
        text_boolean_mask == 1,
        torch.tensor(0.0, device=device),
        torch.tensor(float("-inf"), device=device),
    )
    att_weight_txt, _, _ = model.fdt.extract_txt_sd_ft(text_patch_embeds, pad_mask=pad_mask, return_token_att=True)

    return att_weight_img[:, :196, :], att_weight_txt


def render_overlay(image_path, att_weight_txt, att_weight_img, text_input_ids, tokenizer, words, zoom_level):
    word_ids = text_input_ids[0].cpu().tolist()
    heatmaps = []
    for word in words:
        matches = [
            i for i in range(len(word_ids))
            if word in tokenizer.convert_ids_to_tokens(word_ids[i])
        ]
        if not matches:
            raise ValueError(f"Word '{word}' not found in tokenized caption.")
        index = matches[0]
        codebook_att = att_weight_txt[0, index, :]
        best_code = torch.argmax(codebook_att, dim=-1)
        heatmap = att_weight_img[0, :, best_code].reshape(14, 14).unsqueeze(0).unsqueeze(0)
        heatmap = torch.nn.functional.interpolate(heatmap, (224, 224), mode="bilinear").squeeze(0).detach().cpu()
        heatmaps.append(heatmap)

    heatmap = torch.cat(heatmaps).mean(dim=0).numpy()

    transform = transforms.Compose([
        transforms.CenterCrop(300 * zoom_level),
        transforms.Resize(224),
    ])
    orig_image = transform(Image.open(image_path).convert("RGB"))
    orig_image_np = np.array(orig_image)

    normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-9)
    colored = plt.get_cmap("jet")(normalized)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(orig_image_np, cmap="gray", alpha=1)
    ax.imshow(colored, alpha=0.5)
    ax.set_title("Activation for: " + " ".join(words))
    ax.axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=200)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def build_coord_picker_map():
    map_ = folium.Map(
        location=[31.3879, -115.1367],
        zoom_start=5,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        overlay=False,
        control=True,
    )
    plugins.LocateControl(auto_start=False).add_to(map_)
    map_.get_root().html.add_child(folium.Element(
        """
        <script>
      window.addEventListener('load', function() {
    setTimeout(function() {
        try {
            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    var popup = document.querySelector('.leaflet-popup-content');
                    if (popup) {
                        var coords = popup.textContent.match(/(-?\\d+\\.\\d+)/g);
                        if (coords && coords.length >= 2) {
                            var parentDoc = window.parent.document;
                            var textboxes = parentDoc.querySelectorAll('textarea[data-testid="textbox"]');
                            if (textboxes.length >= 2) {
                                textboxes[0].value = coords[0];
                                textboxes[1].value = coords[1];
                                ['input', 'change'].forEach(eventType => {
                                    textboxes[0].dispatchEvent(new Event(eventType, { bubbles: true }));
                                    textboxes[1].dispatchEvent(new Event(eventType, { bubbles: true }));
                                });
                            }
                        }
                    }
                });
            });
            var popupPane = document.querySelector('.leaflet-popup-pane');
            if (popupPane) {
                observer.observe(popupPane, {childList: true, subtree: true, characterData: true});
            }
        } catch (e) {}
    }, 5000);
});
        </script>
        """
    ))
    map_.add_child(folium.LatLngPopup())
    return map_._repr_html_()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    api_key = load_api_key(bingmap_api)
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(sat2sound_ckpt):
        raise FileNotFoundError(
            f"Sat2Sound checkpoint not found at {sat2sound_ckpt}. "
            "Set SAT2SOUND_CKPT in your environment."
        )

    hparams, model = load_model(sat2sound_ckpt, device)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    map_html = build_coord_picker_map()

    def run(latitude, longitude, zoom_level, text_query, words, use_llava):
        if not latitude or not longitude:
            raise gr.Error("Enter or click a location to pick latitude/longitude.")

        tile_path = os.path.join(log_dir, "map_demo.jpeg")
        download_satellite_tile(
            latitude, longitude, api_key, tile_path,
            size_px=SIZE_SCALE[int(zoom_level)],
        )

        if use_llava or not text_query.strip():
            text_query = make_llava_caption(tile_path, device)

        words_list = [w for w in words.split("_") if w] or [text_query.split()[0]]

        batch, text_input_ids = prepare_batch(
            tile_path, text_query, hparams, tokenizer, device,
            int(zoom_level), float(latitude), float(longitude),
        )
        att_weight_img, att_weight_txt = compute_heatmap(
            model, hparams, batch, sat_type="bingmap", device=device,
        )
        overlay = render_overlay(
            tile_path, att_weight_txt, att_weight_img, text_input_ids,
            tokenizer, words_list, int(zoom_level),
        )
        return overlay, text_query

    with gr.Blocks() as interface:
        gr.Markdown("# Sat2Sound: Fine-grained soundscape heatmap")
        gr.Markdown(
            "Pick a location, enter a soundscape description (or let LLaVA write "
            "one), and choose which word to render an attention heatmap for. "
            "The heatmap shows where in the tile the model believes that word's "
            "sound is most present."
        )
        with gr.Row():
            with gr.Column():
                gr.HTML(value=map_html)
                latitude = gr.Textbox(label="Latitude", interactive=True)
                longitude = gr.Textbox(label="Longitude", interactive=True)
                zoom_level = gr.Radio(choices=[1, 3, 5], value=5, label="Zoom level")
                text_query = gr.Textbox(
                    label="Soundscape description",
                    placeholder="e.g. 'We hear birds and a running river in the forest.'",
                    lines=3,
                )
                words = gr.Textbox(
                    label="Word / phrase to visualize (use '_' to join multiple words)",
                    value="river",
                )
                use_llava = gr.Checkbox(label="Generate caption with LLaVA (requires GPU)", value=False)
                submit = gr.Button("Visualize", variant="primary")
            with gr.Column():
                out_image = gr.Image(label="Attention heatmap overlay", type="pil")
                out_caption = gr.Textbox(label="Caption used")

        submit.click(
            fn=run,
            inputs=[latitude, longitude, zoom_level, text_query, words, use_llava],
            outputs=[out_image, out_caption],
        )

    interface.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
