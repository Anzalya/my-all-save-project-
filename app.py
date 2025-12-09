import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import time
import tempfile

# ----------------- Настройка Streamlit -----------------
st.set_page_config(page_title="AI Ван Гог", layout="centered")
st.title("🎨 AI-Художник — стиль Ван Гога")
st.write("Загрузите фото и картину Ван Гога. Скрипт работает на CPU или GPU автоматически.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Утилиты -----------------
def load_image_pil(pil_image, size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(pil_image).unsqueeze(0).to(device)

def tensor_to_pil(tensor):
    t = tensor.clone().detach().cpu().squeeze(0)
    return transforms.ToPILImage()(t.clamp(0,1))

# Normalization для VGG
cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean.view(-1,1,1)
        self.std = std.view(-1,1,1)
    def forward(self, img):
        return (img - self.mean) / self.std

def get_features(cnn, x, content_layers, style_layers):
    features = {}
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            name = str(layer)
        x = layer(x)
        if name in content_layers:
            features[name] = x
        if name in style_layers:
            features[name] = x
    return features

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

# ----------------- UI -----------------
content_file = st.file_uploader("Загрузите фото (контент)", type=["jpg","jpeg","png"])
style_file = st.file_uploader("Загрузите картину Ван Гога (стиль)", type=["jpg","jpeg","png"])

cols = st.columns(3)
with cols[0]:
    size = st.number_input("Размер (px)", min_value=64, max_value=512, value=256, step=64)
with cols[1]:
    steps = st.number_input("Шагов (итераций)", min_value=20, max_value=1000, value=150, step=10)
with cols[2]:
    lr = st.number_input("LR (Adam)", min_value=0.001, max_value=1.0, value=0.02, step=0.01, format="%.3f")

if content_file and style_file:
    try:
        content_pil = Image.open(content_file).convert("RGB")
        style_pil = Image.open(style_file).convert("RGB")
    except Exception as e:
        st.error("Не удалось открыть одно из изображений: " + str(e))
        st.stop()

    st.image([content_pil, style_pil], caption=["Контент","Стиль"], use_container_width=True)

    if st.button("Применить стиль Ван Гога"):
        st.info("Запускаю перенос стиля... (подождите)")
        start_time = time.time()

        # Преобразуем изображения
        content = load_image_pil(content_pil, size)
        style = load_image_pil(style_pil, size)

        # Загружаем VGG19
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        normalization = Normalization(cnn_mean, cnn_std).to(device)

        # Слои
        content_layers = ['conv_4']
        style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

        content_features = get_features(vgg, normalization(content), content_layers, style_layers)
        style_features = get_features(vgg, normalization(style), content_layers, style_layers)
        style_grams = {layer: gram_matrix(style_features[layer]).detach() for layer in style_layers}

        # Генерируемое изображение
        generated = content.clone().requires_grad_(True).to(device)
        optimizer = optim.Adam([generated], lr=float(lr))

        progress_bar = st.progress(0)
        status = st.empty()
        preview = st.empty()

        style_weight = 1e6
        content_weight = 1.0

        for i in range(1, int(steps)+1):
            optimizer.zero_grad()
            gen_features = get_features(vgg, normalization(generated), content_layers, style_layers)

            content_loss = nn.functional.mse_loss(gen_features['conv_4'], content_features['conv_4']) * content_weight
            style_loss = 0.0
            for layer in style_layers:
                G = gram_matrix(gen_features[layer])
                A = style_grams[layer]
                style_loss += nn.functional.mse_loss(G, A)
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward(retain_graph=True)  # 🔑 фикс ошибки
            optimizer.step()

            if i % max(1, int(steps/10)) == 0 or i == int(steps):
                elapsed = time.time() - start_time
                status.text(f"Итерация {i}/{steps} — Loss: {total_loss.item():.2e} — Время: {elapsed:.1f}s")
                preview.image(tensor_to_pil(generated), caption=f"Preview (iter {i})", use_container_width=True)
                progress_bar.progress(int(i/steps*100))

        result_pil = tensor_to_pil(generated)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            result_pil.save(tmp.name)
            tmp_path = tmp.name

        st.success(f"Готово! Время: {time.time()-start_time:.1f}s")
        st.image(result_pil, caption="Результат", use_container_width=True)
        st.download_button("⬇ Скачать результат", open(tmp_path, "rb"), file_name="vangogh_result.jpg")
else:
    st.info("Загрузите оба изображения (контент и стиль). Рекомендуемый размер 128–256 для быстрого результата.")
