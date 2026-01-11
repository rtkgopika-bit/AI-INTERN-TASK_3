import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
def build_feature_extractor():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(model.children())[:-1]  # remove final layer
    model = nn.Sequential(*modules)
    model.eval()
    return model

feature_extractor = build_feature_extractor()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
def extract_features(image_path):
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img)
    return features.reshape(1, -1)
word2idx = {
    "<start>": 0, "a": 1, "man": 2, "dog": 3, "on": 4,
    "grass": 5, "sitting": 6, "with": 7, "ball": 8, "<end>": 9
}

idx2word = {v: k for k, v in word2idx.items()}

vocab_size = len(word2idx)
class CaptionModel(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=10):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.img_fc = nn.Linear(2048, hidden_size)

    def forward(self, image_vec, caption_tokens):
        embedded = self.embed(caption_tokens)
        h0 = torch.tanh(self.img_fc(image_vec)).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out)
        return out

model = CaptionModel()

def generate_caption(model, image_vec, max_len=10):
    model.eval()
    
    caption = ["<start>"]

    for _ in range(max_len):
        token_ids = torch.tensor([[word2idx[w] for w in caption]])
        outputs = model(image_vec, token_ids)
        next_word_id = outputs[0, -1].argmax().item()
        next_word = idx2word[next_word_id]
        
        if next_word == "<end>":
            break
        
        caption.append(next_word)

    return " ".join(caption[1:])  # remove <start>

def run(image_path):
    print("Extracting features...")
    img_vec = extract_features(image_path)

    print("Generating caption...")
    caption = generate_caption(model, img_vec)

    print("\nGenerated Caption:")
    print("-----------------------")
    print(caption)
    print("-----------------------")

if __name__ == "__main__":
    img = input("Enter image path: ")
    run(img)
