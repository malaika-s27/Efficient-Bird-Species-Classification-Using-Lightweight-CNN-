import numpy as np

# Only include classes your model was trained on (29 classes)
class_map = {
    "barswa": "Barn Swallow",
    "comsan": "Common Sandpiper",
    "eaywag1": "Eastern Yellow Wagtail",
    "thrnig1": "Thornbill Nigricollis",
    "wlwwar": "Willow Warbler",
    "woosan": "Wood Sandpiper",
    "combuz1": "Common Buzzard",
    "eubeat1": "European Beatle",
    "hoopoe": "Hoopoe",
    "cohmar1": "Common Harrier",
    "litegr": "Little Grebe",
    "combul2": "Common Bulbul",
    "rbsrob1": "Rose-Ringed Parakeet",
    "blakit1": "Black Kite",
    "greegr": "Green Grosbeak",
    "gnbcam2": "Great Northern Bird Camera",
    "rerswa1": "Red-Rumped Swallow",
    "somgre1": "Some Green Warbler",
    "colsun2": "Collared Sunbird",
    "ratcis1": "Rattle Cisticola",
    "blbpuf2": "Blue Puffin",
    "categr": "Cattle Egret",
    "tafpri1": "Taffy Prince Bird",
    "carcha1": "Caracara",
    "egygoo": "Egyptian Goose",
    "grecor": "Great Cormorant",
    "fotdro5": "Forest Drosera",
    "gargan": "Garden Warbler",
    "yertin1": "Yellow Robin"
}

# Save dictionary as a .npy file
np.save("class_map.npy", class_map)
print("class_map.npy saved successfully!")
