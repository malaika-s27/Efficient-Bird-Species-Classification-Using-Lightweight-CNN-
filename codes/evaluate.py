import numpy as np, torch
from torch.utils.data import DataLoader
from dataset import MelDataset
from model import build_efficientnet_b0
import config
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Needed for better heatmap labels
import os

# --- Your existing model loading and prediction code ---
# Load stage2 model
class_map = np.load(config.CLASS_MAP, allow_pickle=True)
num_cls = len(class_map)
model = build_efficientnet_b0(num_classes=num_cls)
model.load_state_dict(torch.load(config.MODEL_OUT_STAGE2, map_location='cpu'))
model.eval()

ds = MelDataset(processed_dir=config.PROCESSED_DIR, train=False, augment=False)
loader = DataLoader(ds, batch_size=config.BATCH_SIZE)

y_true=[]; y_pred=[]
with torch.no_grad():
    for x,y in loader:
        out = model(x)
        preds = out.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

# --- Generate and print classification report (Text output) ---
print("Classification report:")
print(classification_report(y_true, y_pred, target_names=class_map))

# --- Generate Confusion Matrix (Raw data) ---
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape:", cm.shape)
np.save(os.path.join(os.path.dirname(config.MODEL_OUT_STAGE2),"confusion_matrix.npy"), cm)
print("Saved confusion matrix.")

# --- Code to plot the Confusion Matrix (Visual Output) ---
def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, filename="confusion_matrix.png"):
    """
    Generates and saves a heatmap visualization of the confusion matrix.
    """
    # Create a DataFrame for better visualization with class names
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    plt.figure(figsize=(15, 12)) # Adjust size for many classes
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues') # Use 'd' format for integer counts
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix Heatmap')
    plt.xticks(rotation=45, ha='right') # Rotate labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved confusion matrix plot to {filename}")
    plt.show() # Display plot window

# Run the plotting function
plot_confusion_matrix_heatmap(y_true, y_pred, class_map, filename="confusion_matrix_plot.png")
