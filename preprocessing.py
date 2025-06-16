#Author: Michael Schmidt / Islam Elmaaroufi

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
# Standard libraries für ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Computer Vision
import albumentations as A
import cv2


class EfficientTrafficSignPreprocessor:
    """
    Schlanker Preprocessor mit Standard-Bibliotheken für Verkehrsschilder-Daten.
    """
    
    def __init__(self, 
                 data_dir: str,
                 target_size: Tuple[int, int] = (224, 224),
                 normalization: str = 'imagenet'):
        """
        Args:
            data_dir: Pfad zum Datenverzeichnis
            target_size: Zielgröße (height, width)
            normalization: 'imagenet', 'minmax', oder 'none'
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalization = normalization
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Albumentations Transform Pipeline
        self._setup_transforms()
        
        # Label Encoder für Klassen
        self.label_encoder = LabelEncoder()
    
    def _setup_transforms(self):
        """Erstellt optimierte Transform-Pipeline mit Albumentations."""
        transforms = [
            # Automatisches Resize mit Aspect Ratio
            A.LongestMaxSize(max_size=max(self.target_size)),
            A.PadIfNeeded(
                min_height=self.target_size[0], 
                min_width=self.target_size[1],
                border_mode=cv2.BORDER_CONSTANT, 
                #value=0
            )
        ]
        
        # Normalisierung hinzufügen
        if self.normalization == 'imagenet':
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        elif self.normalization == 'minmax':
            transforms.append(A.Normalize(mean=0, std=255))  # /255
        
        self.transform = A.Compose(transforms)
    
    def _find_image_files(self) -> List[Tuple[Path, str]]:
        """
        Findet alle Bilddateien rekursiv und extrahiert Label aus Ordnernamen.
        
        Returns:
            List von (image_path, label) Tupeln
        """
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        # Durchsuche alle Unterordner
        for root, dirs, files in os.walk(self.data_dir):
            root_path = Path(root)
            
            # Label aus Ordnernamen extrahieren (Text nach Unterstrich)
            if root_path != self.data_dir:  # Nicht root directory
                folder_name = root_path.name
                # Extrahiere Text nach Unterstrich (z.B. "5_StopSchild" -> "StopSchild")
                if '_' in folder_name:
                    label = folder_name.split('_', 1)[1]  # Nimm alles nach dem ersten Unterstrich
                else:
                    label = folder_name  # Falls kein Unterstrich, nimm ganzen Namen
            else:
                label = 'unknown'  # Falls Bilder direkt im root liegen
            
            # Alle Bilddateien in diesem Ordner finden
            for file in files:
                file_path = Path(file)
                if file_path.suffix.lower() in extensions:
                    full_path = root_path / file
                    image_files.append((full_path, label))
        
        self.logger.info(f"Gefunden: {len(image_files)} Bilddateien")
        return image_files
    
    def _load_image_with_metadata(self, image_path: Path, label: str) -> Tuple[np.ndarray, Dict, str]:
        """
        Lädt Bild und optional JSON-Metadaten.
        
        Args:
            image_path: Pfad zum Bild
            label: Bereits extrahiertes Label aus Ordnernamen
            
        Returns:
            (image_array, metadata_dict, label)
        """
        try:
            # Bild laden (BGR -> RGB)
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Bild konnte nicht geladen werden: {image_path}")
                return None, None, None
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # JSON-Metadaten laden (optional)
            metadata = {}
            json_path = image_path.with_suffix('.json')
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    self.logger.warning(f"JSON-Metadaten konnten nicht geladen werden für {image_path}: {e}")
                    metadata = {}
            
            # Label direkt weitergeben (bereits in _find_image_files extrahiert)
            return image, metadata, label
            
        except Exception as e:
            self.logger.error(f"Fehler bei {image_path}: {e}")
            return None, None, None
    
    def process_dataset(self, 
                       output_dir: str,
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """
        Verarbeitet komplettes Dataset mit Standard-Bibliotheken.
        
        Args:
            output_dir: Ausgabeverzeichnis
            test_size: Test-Split Anteil
            val_size: Validation-Split Anteil  
            random_state: Seed für Reproduzierbarkeit
            
        Returns:
            Dictionary mit Verarbeitungsstatistiken
        """
        self.logger.info("Starte effiziente Dataset-Verarbeitung...")
        
        # Output-Verzeichnis
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Alle Bilddateien mit Labels finden
        image_files_with_labels = self._find_image_files()
        
        # Daten laden und transformieren
        images, labels, metadata_list = self._load_and_transform_data(image_files_with_labels)
        
        if len(images) == 0:
            raise ValueError("Keine Bilder konnten verarbeitet werden!")
        
        # Labels encodieren (String -> Integer)
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Train/Val/Test Split mit sklearn (stratifiziert)
        splits = self._create_stratified_splits(
            np.array(images), encoded_labels, metadata_list,
            test_size, val_size, random_state
        )
        
        # Daten speichern
        self._save_processed_data(splits, output_path)
        
        # Statistiken
        stats = {
            'total_processed': len(images),
            'unique_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'image_shape': np.array(images).shape[1:],
            'normalization': self.normalization,
            'splits': {
                'train': len(splits['X_train']),
                'val': len(splits['X_val']), 
                'test': len(splits['X_test'])
            }
        }
        
        # Statistiken speichern
        with open(output_path / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info("Verarbeitung abgeschlossen!")
        return stats
    
    def _load_and_transform_data(self, image_files_with_labels: List[Tuple[Path, str]]) -> Tuple[List, List, List]:
        """Lädt und transformiert alle Bilder effizient."""
        images, labels, metadata_list = [], [], []
        
        for img_path, label in tqdm(image_files_with_labels, desc="Bilder verarbeiten"):
            image, metadata, extracted_label = self._load_image_with_metadata(img_path, label)
                
            # Skip fehlerhafte Bilder oder unbekannte Labels
            if image is None or extracted_label == 'unknown':
                self.logger.warning(f"Überspringe {img_path}: Bild={image is not None}, Label={extracted_label}")
                continue
                
            # Albumentations Transform anwenden
            try:
                transformed = self.transform(image=image)
                processed_image = transformed['image']
               
                processed_image = np.transpose(processed_image, (2, 0, 1))  
                processed_image = processed_image.astype(np.float32)
                images.append(processed_image)
                labels.append(extracted_label)
                metadata_list.append(metadata)
                
            except Exception as e:
                self.logger.error(f"Transform-Fehler bei {img_path}: {e}")
                continue
        
        self.logger.info(f"Erfolgreich verarbeitet: {len(images)} Bilder")
        return images, labels, metadata_list
    
    def _create_stratified_splits(self, X: np.ndarray, y: np.ndarray, metadata: List,
                                test_size: float, val_size: float, random_state: int) -> Dict:
        """Erstellt stratifizierte Splits mit sklearn."""
        
        # Prüfen, ob genügend Samples pro Klasse vorhanden sind
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples = np.min(class_counts)
        
        if min_samples < 2:
            self.logger.warning("Mindestens eine Klasse hat nur 1 Sample - verwende einfache Splits ohne Stratifizierung")
            # Fallback auf normale Splits
            X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
                X, y, metadata, 
                test_size=test_size, 
                random_state=random_state
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                X_temp, y_temp, meta_temp,
                test_size=val_size_adjusted,
                random_state=random_state
            )
        else:
            # Stratifizierte Splits
            X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
                X, y, metadata, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
                X_temp, y_temp, meta_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=y_temp
            )
        
        return {
            'X_train': X_train, 'y_train': y_train, 'meta_train': meta_train,
            'X_val': X_val, 'y_val': y_val, 'meta_val': meta_val,
            'X_test': X_test, 'y_test': y_test, 'meta_test': meta_test
        }
    
    def _save_processed_data(self, splits: Dict, output_path: Path):
         """Speichert verarbeitete Daten einzeln als numpy arrays."""
         self.logger.info("Speichere verarbeitete Daten in Blöcken...")

         for key in ['X_train', 'y_train', 'meta_train',
                'X_val', 'y_val', 'meta_val',
                'X_test', 'y_test', 'meta_test']:
          array = splits[key]

        # In Blöcken speichern, falls Liste (z.B. Metadaten)
          file_path = output_path / f"{key}.npy"

          try:
            if isinstance(array, list):
                # Speichere Listen (z. B. Metadaten) als Pickle-Dateien
                with open(file_path.with_suffix(".pkl"), 'wb') as f:
                    pickle.dump(array, f)
            else:
                np.save(file_path, array)
            self.logger.info(f"{key} gespeichert: {file_path}")
          except Exception as e:
            self.logger.error(f"Fehler beim Speichern von {key}: {e}")

    # Label Encoder separat speichern
         with open(output_path / 'label_encoder.pkl', 'wb') as f:
          pickle.dump(self.label_encoder, f)



def main():
    """
    Hauptfunktion - Einfache Konfiguration
    """
    
    # ========== KONFIGURATION ==========
    DATA_DIR = "/Users/mac/Desktop/Ogre_subsample"
    OUTPUT_DIR = "/Users/mac/Desktop/Nachher"
    
    TARGET_SIZE = (224, 224)           # Standard für viele CNN Modelle
    NORMALIZATION = 'imagenet'         # 'imagenet', 'minmax', oder 'none'
    
    TEST_SIZE = 0.2                    # 20% Test
    VAL_SIZE = 0.2                     # 20% Validation  
    RANDOM_STATE = 42                  # Reproduzierbarkeit
    
    # ====================================
    
    # Preprocessor erstellen
    preprocessor = EfficientTrafficSignPreprocessor(
        data_dir=DATA_DIR,
        target_size=TARGET_SIZE,
        normalization=NORMALIZATION
    )
    
    # Dataset verarbeiten
    try:
        stats = preprocessor.process_dataset(
            output_dir=OUTPUT_DIR,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Ergebnisse
        print("\n" + "="*50)
        print("VERARBEITUNGS-STATISTIKEN")
        print("="*50)
        print(f"Verarbeitete Bilder: {stats['total_processed']:,}")
        print(f"Klassen: {stats['unique_classes']}")
        print(f"Klassen-Namen: {', '.join(stats['class_names'][:10])}...")  # Erste 10
        print(f"Bild-Shape: {stats['image_shape']}")
        print(f"Train: {stats['splits']['train']:,} | Val: {stats['splits']['val']:,} | Test: {stats['splits']['test']:,}")
        print(f"Normalisierung: {stats['normalization']}")
        
    except Exception as e:
        print(f"Fehler: {e}")


if __name__ == "__main__":
    main()
