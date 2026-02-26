import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json

# Configuration de la page
st.set_page_config(page_title="Plant Doctor", page_icon="🌿")
st.title("🌿 Diagnostic des Maladies de Plantes")

# 1. Charger les noms de classes
@st.cache_data
def get_classes():
    with open('class_names.json', 'r') as f:
        return json.load(f)

class_names = get_classes()

# 2. Définir l'architecture (Doit être identique au notebook)
# Note: On utilise ici l'architecture simplifiée PlantCNN
class PlantCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Charger le modèle
@st.cache_resource
def load_model():
    model = PlantCNN(num_classes=len(class_names))
    # map_location='cpu' est crucial car le serveur n'a pas de GPU
    model.load_state_dict(torch.load("best_plant_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 4. Interface utilisateur
img_file = st.camera_input("Prendre une photo de la feuille")

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Photo capturée", use_column_width=True)
    
    # Transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    
        top_probs, top_indices = probs.topk(3)

    # --- SECTION AFFICHAGE (Le résultat) ---
    st.subheader("🔍 Analyse détaillée :")
    
    # Remplace ta boucle for par celle-ci :
    st.subheader("🔍 Analyse détaillée :")
    
    # On vérifie si la meilleure confiance est trop basse
    if top_probs[0].item() < 0.50:
        st.warning("⚠️ L'IA a un doute. Le diagnostic ci-dessous est peu fiable. Essayez de prendre la photo de plus près ou avec un meilleur éclairage.")

    for i in range(3):
        label = class_names[top_indices[i]]
        score = top_probs[i].item() * 100
        
        st.write(f"**{label}**")
        st.progress(score / 100)
        st.write(f"Confiance : {score:.1f}%")
        st.divider()