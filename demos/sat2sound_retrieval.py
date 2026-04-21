"""Gradio demo: retrieve a synthetic soundscape from a gallery given a satellite image.

Workflow:
  1. User clicks a location on an interactive Folium world map.
  2. We download a satellite tile from Azure Maps (using BingMap-style credentials).
  3. We run the trained Sat2Sound model to produce a satellite embedding.
  4. We cosine-match the embedding against pre-computed text embeddings in the
     gallery HDF5 and take the top-1 caption.
  5. We play back the pre-synthesized audio stored alongside the top-1 caption.

Required environment (see demos/demo_config.py):

    SAT2SOUND_CKPT         path to the trained Sat2Sound checkpoint
    SAT2SOUND_GALLERY      path to the gallery HDF5 (has audio_embedding,
                           text_embedding_zl{1,3,5}, llava_caption_zl{1,3,5},
                           synth_audio_zl{1,3,5}, audio_raw, audio_caption)
    BINGMAP_API_KEY        Bing/Azure Maps key (or put it in .secrets/bingmap_api.txt)

Run: ``python demos/sat2sound_retrieval.py``
"""

import os
import random
import time
import urllib.request
from argparse import Namespace

import folium
import gradio as gr
import h5py
import numpy as np
import torch
import torchaudio
from folium import plugins
from PIL import Image
from torchvision import transforms

from demos.demo_config import bingmap_api, gallery_path, log_dir, metadata_config, sat2sound_ckpt
from src.engine import l2normalize, sat2soundModel
from utilities.utils import sat_transform


zoom_levels_dict = {300: 1, 900: 3, 1500: 5}


def load_api_key(file_path):
    """Load the Bing/Azure Maps key from file, falling back to the env var."""
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


def load_gallery(path):
    gallery = h5py.File(path, "r")
    sample_ids = [sid.decode() for sid in gallery["sample_id"][:]]
    zl_captions = {
        zl: [cap.decode() for cap in gallery[f"llava_caption_zl{zl}"][:]]
        for zl in [1, 3, 5]
    }
    audio_embeddings = torch.Tensor(np.stack([emb for emb in gallery["audio_embedding"][:]]).T)
    zl_text_embeddings = {
        zl: torch.Tensor(np.stack([emb for emb in gallery[f"text_embedding_zl{zl}"][:]]).T)
        for zl in [1, 3, 5]
    }
    print("Gallery loaded")
    return {
        "handle": gallery,
        "sample_ids": sample_ids,
        "zl_captions": zl_captions,
        "audio_embeddings": audio_embeddings,
        "zl_text_embeddings": zl_text_embeddings,
    }


def get_sample(gallery_handle, idx, zl):
    return {
        "sample_id": gallery_handle["sample_id"][idx].decode(),
        "llava_caption": gallery_handle[f"llava_caption_zl{zl}"][idx].decode(),
        "audio_caption": gallery_handle["audio_caption"][idx].decode(),
        "raw_audio": gallery_handle["audio_raw"][idx],
        "synth_audio": gallery_handle[f"synth_audio_zl{zl}"][idx],
    }


def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model(ckpt_path, device):
    pretrained_ckpt = torch.load(ckpt_path, map_location=device)
    hparams = pretrained_ckpt["hyper_parameters"]
    hparams["meta_droprate"] = 0.0
    model = sat2soundModel(Namespace(**hparams)).to(device)
    model.load_state_dict(pretrained_ckpt["state_dict"], strict=False)
    model.eval()
    for params in model.parameters():
        params.requires_grad = False
    return Namespace(**hparams), model


def download(url, out_file):
    try:
        urllib.request.urlretrieve(url, out_file)
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code)
        raise
    except Exception as e:
        print("Other Error:", e)
        raise


def process_input(image_path, device, zoom_level):
    sat_tr = sat_transform(is_train=False, input_size=224, sat_type="bingmap", zoom_level=zoom_level)
    image = Image.open(image_path)
    final_image = sat_tr(image).unsqueeze(0).to(device)
    return {"sat": final_image, "zoom_level": float(zoom_level)}


def prepare_metadata(lat, lon):
    lat = float(str(lat).strip())
    lon = float(str(lon).strip())
    audio_source_map = {"yfcc": 0, "iNat": 1, "aporee": 2, "freesound": 3}
    caption_source_map = {"meta": 0, "qwen": 1, "pengi": 2}
    metadata = {}
    metadata["audio_source"] = torch.tensor(audio_source_map[metadata_config.audio_source]).long().unsqueeze(0)
    metadata["caption_source"] = torch.tensor(caption_source_map[metadata_config.caption_source]).long().unsqueeze(0)
    metadata["latlong"] = torch.tensor([
        np.sin(np.pi * lat / 90), np.cos(np.pi * lat / 90),
        np.sin(np.pi * lon / 180), np.cos(np.pi * lon / 180),
    ]).float().unsqueeze(0)
    metadata["time"] = torch.tensor([
        np.sin(2 * np.pi * metadata_config.time / 23),
        np.cos(2 * np.pi * metadata_config.time / 23),
    ]).float().unsqueeze(0)
    metadata["time_valid"] = torch.tensor(True).long().unsqueeze(0)
    metadata["month"] = torch.tensor([
        np.sin(2 * np.pi * metadata_config.month / 12),
        np.cos(2 * np.pi * metadata_config.month / 12),
    ]).float().unsqueeze(0)
    metadata["month_valid"] = torch.tensor(True).long().unsqueeze(0)
    return metadata


def get_sat_embedding(model, hparams, sat_image, zoom_level, lat, lon):
    metadata = prepare_metadata(lat, lon)
    sat_token_embeddings = model.satmae_backbone(sat_image, zoom_level=[zoom_level], sat_type=hparams.sat_type)
    device = sat_token_embeddings.device
    if hparams.metadata_type != "none":
        sat_token_embeddings_before_fdt = model.sat_encoder(
            sat_embeddings=sat_token_embeddings,
            audio_source=metadata["audio_source"].to(device),
            caption_source=metadata["caption_source"].to(device),
            latlong=metadata["latlong"].to(device),
            time=metadata["time"].to(device),
            month=metadata["month"].to(device),
            time_valid=metadata["time_valid"].to(device),
            month_valid=metadata["month_valid"].to(device),
        )
    else:
        sat_token_embeddings_before_fdt = sat_token_embeddings
    _, sd_img_ft, _ = model.fdt.extract_img_sd_ft(sat_token_embeddings_before_fdt, return_token_att=False)
    return sd_img_ft


def download_satellite_tile(lat, lon, api_key, out_file, zoom=18, width=1500, height=1500):
    # Azure Maps static imagery endpoint (compatible with Bing-style keys).
    url = (
        f"https://atlas.microsoft.com/map/static"
        f"?api-version=2024-04-01"
        f"&tilesetId=microsoft.imagery"
        f"&center={lon},{lat}"
        f"&zoom={zoom}"
        f"&width={width}"
        f"&height={height}"
        f"&subscription-key={api_key}"
    )
    download(url=url, out_file=out_file)


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
    # Click-to-pick-coord JS: when the user clicks on the map, Folium pops up
    # lat/lon in a popup. This observer copies those into the two Gradio
    # textboxes in the parent document so the user doesn't have to retype them.
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
        } catch (e) {
            console.error('Error in map setup:', e);
        }
    }, 5000);
});
        </script>
        """
    ))
    map_.add_child(folium.LatLngPopup())
    plugins.LocateControl(auto_start=False).add_to(map_)
    return map_._repr_html_()


def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    api_key = load_api_key(bingmap_api)

    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(sat2sound_ckpt):
        raise FileNotFoundError(
            f"Sat2Sound checkpoint not found at {sat2sound_ckpt}. "
            "Set SAT2SOUND_CKPT in your environment or place the file at the default path."
        )
    if not os.path.exists(gallery_path):
        raise FileNotFoundError(
            f"Gallery HDF5 not found at {gallery_path}. "
            "Set SAT2SOUND_GALLERY in your environment."
        )

    hparams, model = load_model(ckpt_path=sat2sound_ckpt, device=device)
    gallery = load_gallery(gallery_path)
    map_html = build_coord_picker_map()

    def get_audio(html, latitude, longitude, sat_img_height):
        init_time = time.time()
        out_file = os.path.join(log_dir, "demo.jpeg")
        download_satellite_tile(latitude, longitude, api_key, out_file)
        print("Satellite image downloaded")

        transform = transforms.Compose([
            transforms.CenterCrop(sat_img_height),
            transforms.Resize(224),
        ])
        orig_image_np = np.array(transform(Image.open(out_file).convert("RGB")))

        sat_data = process_input(out_file, device, zoom_levels_dict[sat_img_height])
        sat_embed = get_sat_embedding(model, hparams, sat_data["sat"], sat_data["zoom_level"], latitude, longitude)

        zl = int(sat_data["zoom_level"])
        text_embeds = l2normalize(gallery["zl_text_embeddings"][zl].T).to(sat_embed.device)
        query_embed = l2normalize(sat_embed)
        cosine_similarities = torch.mm(query_embed, text_embeds.T)
        _, topk_indices = torch.topk(cosine_similarities, 1, dim=1)
        top_idx = topk_indices[0].detach().cpu().tolist()[0]

        caption = gallery["zl_captions"][zl][top_idx]
        top_sample = get_sample(gallery["handle"], top_idx, zl)

        out_audio = out_file.replace("jpeg", "wav")
        torchaudio.save(out_audio, torch.Tensor(top_sample["synth_audio"]), 44100)
        Image.fromarray(orig_image_np).save(out_file, format="JPEG")

        print("Time taken:", time.time() - init_time)
        return out_file, caption, out_audio

    def clear_fields():
        return "", "", 300

    with gr.Blocks() as interface:
        gr.Markdown("# Sat2Sound: Retrieval Demo")
        gr.Markdown("Pick a location on the map. We retrieve a matching soundscape caption and play its pre-synthesized audio.")
        with gr.Row():
            with gr.Column():
                gr.HTML(value=map_html)
                latitude = gr.Textbox(label="Latitude", interactive=True)
                longitude = gr.Textbox(label="Longitude", interactive=True)
                sat_img_height = gr.Slider(
                    label="Satellite image field of view (px)",
                    minimum=300, maximum=1500, step=600, value=300,
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear Fields")
                    submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                sat_img = gr.Image(label="Satellite Image", type="filepath")
                caption = gr.Textbox(label="Top-1 LLaVA caption retrieved by the Sat2Sound model")
                output = gr.Audio(label="Synthetic soundscape", type="filepath")

        clear_btn.click(fn=clear_fields, inputs=None, outputs=[latitude, longitude, sat_img_height])
        submit_btn.click(
            fn=get_audio,
            inputs=[gr.State(None), latitude, longitude, sat_img_height],
            outputs=[sat_img, caption, output],
        )

    interface.launch(share=True, debug=True)


if __name__ == "__main__":
    main()
