"""
Facial Mood Detection Using OpenCV
CK+48 Dataset Implementation
Complete Project Code
"""

import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

class CKPlusDataLoader:
    """Loads and preprocesses the CK+48 dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        # Emotion labels for CK+48 (7 emotions)
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
    def load_ck48_dataset(self):
        """
        Load CK+48 dataset
        Expected structure:
        dataset_path/
            train/
                angry/
                    img1.jpg
                disgust/
                ...
            test/
                angry/
                ...
        OR
        dataset_path/
            angry/
                img1.jpg
            disgust/
            ...
        """
        print("Loading CK+48 dataset...")
        print(f"Dataset path: {os.path.abspath(self.dataset_path)}")
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Check if train/test split exists
        has_train_test = (os.path.exists(os.path.join(self.dataset_path, 'train')) and 
                         os.path.exists(os.path.join(self.dataset_path, 'test')))
        
        if has_train_test:
            print("Found train/test split structure")
            train_images, train_labels = self._load_from_folders(
                os.path.join(self.dataset_path, 'train')
            )
            test_images, test_labels = self._load_from_folders(
                os.path.join(self.dataset_path, 'test')
            )
            return (np.array(train_images), np.array(train_labels), 
                   np.array(test_images), np.array(test_labels))
        else:
            print("Found single folder structure")
            images, labels = self._load_from_folders(self.dataset_path)
            return np.array(images), np.array(labels)
    
    def _load_from_folders(self, base_path):
        """Load images from emotion folders"""
        images = []
        labels = []
        
        # Map folder names to labels (case-insensitive)
        emotion_map = {
            'angry': 0, 'anger': 0,
            'disgust': 1, 'disgusted': 1,
            'fear': 2, 'fearful': 2,
            'happy': 3, 'happiness': 3,
            'sad': 4, 'sadness': 4,
            'surprise': 5, 'surprised': 5,
            'neutral': 6
        }
        
        # Get all subdirectories
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Path does not exist: {base_path}")
            
        folders = sorted([f for f in os.listdir(base_path) 
                         if os.path.isdir(os.path.join(base_path, f))])
        
        if not folders:
            raise ValueError(f"No emotion folders found in {base_path}")
        
        print(f"Found emotion folders: {folders}")
        
        for folder in folders:
            folder_lower = folder.lower()
            if folder_lower not in emotion_map:
                print(f"Warning: Unknown emotion folder '{folder}', skipping...")
                continue
            
            label = emotion_map[folder_lower]
            folder_path = os.path.join(base_path, folder)
            
            # Load all images in this folder
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not image_files:
                print(f"Warning: No images found in {folder}")
                continue
                
            print(f"Loading {len(image_files)} images from {folder}...")
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize if not 48x48
                    if img.shape != (48, 48):
                        img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Warning: Could not load {img_path}")
        
        if not images:
            raise ValueError("No images were loaded! Check your dataset structure.")
            
        print(f"Successfully loaded {len(images)} images total")
        return images, labels
    
    def create_synthetic_dataset(self, num_samples=500):
        """
        Create a synthetic dataset for demonstration
        (Use this if you don't have CK+48 dataset)
        """
        print("Creating synthetic dataset for demonstration...")
        
        images = []
        labels = []
        
        for _ in range(num_samples):
            # Create synthetic face-like images
            img = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
            # Add some structure to simulate faces
            cv2.circle(img, (24, 20), 8, 200, -1)  # Eyes region
            cv2.circle(img, (24, 35), 10, 200, 1)  # Mouth region
            
            images.append(img)
            labels.append(np.random.randint(0, 7))  # 7 emotions
        
        print(f"Created {len(images)} synthetic images")
        return np.array(images), np.array(labels)


# ============================================================================
# PART 2: FACE DETECTION AND PREPROCESSING
# ============================================================================

class FaceDetector:
    """Handles face detection using Haar Cascade"""
    
    def __init__(self):
        # Load pre-trained Haar Cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar Cascade classifier")
        
        print("Face detector initialized successfully")
    
    def detect_face(self, image):
        """
        Detect face in image and return the face ROI
        """
        faces = self.face_cascade.detectMultiScale(
            image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the first (largest) face
            x, y, w, h = faces[0]
            return image[y:y+h, x:x+w], (x, y, w, h)
        return None, None
    
    def preprocess_face(self, face, target_size=(48, 48)):
        """
        Preprocess detected face for model input
        """
        if face is None:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face, target_size)
        
        # Histogram equalization for better contrast
        face_equalized = cv2.equalizeHist(face_resized)
        
        # Normalize pixel values
        face_normalized = face_equalized / 255.0
        
        return face_normalized


# ============================================================================
# PART 3: FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract features from facial images"""
    
    def __init__(self, method='eigenface'):
        self.method = method
        self.pca = None
        self.n_components = 50
        print(f"Feature extractor initialized with method: {method}")
    
    def extract_features(self, images):
        """
        Extract features using specified method
        """
        if self.method == 'eigenface':
            return self._extract_eigenface_features(images)
        elif self.method == 'hog':
            return self._extract_hog_features(images)
        else:
            return self._flatten_images(images)
    
    def _flatten_images(self, images):
        """Simple flattening of images"""
        return images.reshape(len(images), -1)
    
    def _extract_eigenface_features(self, images):
        """Extract features using PCA (Eigenfaces)"""
        # Flatten images
        flattened = self._flatten_images(images)
        
        if self.pca is None:
            # Fit PCA
            n_components = min(self.n_components, flattened.shape[0], flattened.shape[1])
            self.pca = PCA(n_components=n_components)
            print(f"Fitting PCA with {n_components} components...")
            features = self.pca.fit_transform(flattened)
            print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        else:
            features = self.pca.transform(flattened)
        
        return features
    
    def _extract_hog_features(self, images):
        """Extract HOG features"""
        try:
            from skimage.feature import hog
        except ImportError:
            print("Warning: scikit-image not installed. Using flattened features instead.")
            return self._flatten_images(images)
        
        features = []
        
        for img in images:
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            feat = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False)
            features.append(feat)
        
        return np.array(features)


# ============================================================================
# PART 4: MODEL TRAINING
# ============================================================================

class MoodClassifier:
    """Mood classification using OpenCV ML"""
    
    def __init__(self, classifier_type='svm', num_classes=7):
        self.classifier_type = classifier_type
        self.model = None
        self.num_classes = num_classes
        
        # Emotion map for CK+48 (7 emotions)
        self.emotion_map = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear', 
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'
        }
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the classifier"""
        if self.classifier_type == 'svm':
            self.model = cv2.ml.SVM_create()
            self.model.setType(cv2.ml.SVM_C_SVC)
            self.model.setKernel(cv2.ml.SVM_LINEAR)
            self.model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
            print("Initialized SVM classifier")
        elif self.classifier_type == 'knn':
            self.model = cv2.ml.KNearest_create()
            print("Initialized k-NN classifier")
        elif self.classifier_type == 'ann':
            self.model = cv2.ml.ANN_MLP_create()
            print("Initialized ANN classifier")
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def train(self, X_train, y_train):
        """Train the classifier"""
        print(f"Training {self.classifier_type.upper()} classifier...")
        print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Ensure correct data types
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)
        
        if self.classifier_type == 'svm':
            self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        elif self.classifier_type == 'knn':
            self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        elif self.classifier_type == 'ann':
            # Configure ANN layers
            layer_sizes = np.array([X_train.shape[1], 128, 64, self.num_classes])
            self.model.setLayerSizes(layer_sizes)
            self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
            self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
            
            # Convert labels to one-hot encoding for ANN
            y_train_onehot = np.zeros((len(y_train), self.num_classes))
            for i, label in enumerate(y_train):
                y_train_onehot[i, label] = 1
            
            self.model.train(X_train, cv2.ml.ROW_SAMPLE, y_train_onehot.astype(np.float32))
        
        print("Training completed!")
    
    def predict(self, X):
        """Predict emotions for input features"""
        X = X.astype(np.float32)
        
        if self.classifier_type == 'ann':
            _, predictions = self.model.predict(X)
            predictions = np.argmax(predictions, axis=1)
        else:
            _, predictions = self.model.predict(X)
            predictions = predictions.astype(np.int32).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities (approximation for non-probabilistic classifiers)"""
        predictions = self.predict(X)
        
        # Create simple probability distribution
        probs = np.zeros((len(predictions), self.num_classes))
        for i, pred in enumerate(predictions):
            probs[i, pred] = 0.8
            probs[i, :] += 0.2 / self.num_classes
            probs[i, :] /= probs[i, :].sum()
        
        return probs
    
    def save_model(self, filepath):
        """Save trained model"""
        try:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            if self.classifier_type == 'svm':
                self.model = cv2.ml.SVM_load(filepath)
            elif self.classifier_type == 'knn':
                self.model = cv2.ml.KNearest_load(filepath)
            elif self.classifier_type == 'ann':
                self.model = cv2.ml.ANN_MLP_load(filepath)
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")


# ============================================================================
# PART 5: EVALUATION AND VISUALIZATION
# ============================================================================

class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, emotion_labels):
        self.emotion_labels = emotion_labels
    
    def evaluate(self, y_true, y_pred):
        """Evaluate model performance"""
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        
        # Classification report
        print("\nClassification Report:")
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        target_names = [self.emotion_labels[i] for i in unique_labels]
        print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
        
        return cm, accuracy
    
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        labels = [self.emotion_labels[i] for i in sorted(self.emotion_labels.keys())]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Facial Mood Detection', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            print(f"Could not save confusion matrix: {e}")
        
        plt.show()
    
    def plot_sample_predictions(self, images, true_labels, pred_labels, 
                               num_samples=9, save_path='sample_predictions.png'):
        """Plot sample predictions"""
        num_samples = min(num_samples, len(images))
        
        rows = int(np.sqrt(num_samples))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                ax.imshow(images[i], cmap='gray')
                true_emotion = self.emotion_labels[true_labels[i]]
                pred_emotion = self.emotion_labels[pred_labels[i]]
                
                color = 'green' if true_labels[i] == pred_labels[i] else 'red'
                ax.set_title(f'True: {true_emotion}\nPred: {pred_emotion}', 
                           color=color, fontweight='bold', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions saved to {save_path}")
        except Exception as e:
            print(f"Could not save sample predictions: {e}")
        
        plt.show()
    
    def plot_class_distribution(self, y, title="Class Distribution", save_path='class_distribution.png'):
        """Plot class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        labels = [self.emotion_labels[i] for i in unique]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labels, counts, color='steelblue', alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Emotion', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution saved to {save_path}")
        except Exception as e:
            print(f"Could not save class distribution: {e}")
        
        plt.show()


# ============================================================================
# PART 6: COMPLETE PIPELINE
# ============================================================================

class MoodDetectionPipeline:
    """Complete mood detection pipeline"""
    
    def __init__(self, classifier_type='svm', feature_method='eigenface'):
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor(method=feature_method)
        self.classifier = MoodClassifier(classifier_type=classifier_type)
        self.evaluator = ModelEvaluator(self.classifier.emotion_map)
    
    def process_images(self, images):
        """Process images through face detection and preprocessing"""
        processed_faces = []
        valid_indices = []
        
        print("Processing images through face detection...")
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processed {i}/{len(images)} images...")
                
            face, _ = self.face_detector.detect_face(img)
            if face is not None:
                processed = self.face_detector.preprocess_face(face)
                if processed is not None:
                    processed_faces.append(processed)
                    valid_indices.append(i)
        
        print(f"Successfully processed {len(processed_faces)}/{len(images)} images")
        return np.array(processed_faces), valid_indices
    
    def train_pipeline(self, X_train, y_train):
        """Train the complete pipeline"""
        # Extract features
        print("\nExtracting features from training data...")
        X_features = self.feature_extractor.extract_features(X_train)
        print(f"Feature shape: {X_features.shape}")
        
        # Train classifier
        self.classifier.train(X_features, y_train)
        
        return X_features
    
    def evaluate_pipeline(self, X_test, y_test, test_images_original):
        """Evaluate the pipeline"""
        # Extract features
        print("\nExtracting features from test data...")
        X_features = self.feature_extractor.extract_features(X_test)
        
        # Predict
        print("Making predictions...")
        predictions = self.classifier.predict(X_features)
        
        # Evaluate
        cm, accuracy = self.evaluator.evaluate(y_test, predictions)
        
        # Plot results
        print("\nGenerating visualizations...")
        self.evaluator.plot_confusion_matrix(cm)
        
        # Show sample predictions
        if len(test_images_original) > 0:
            sample_size = min(9, len(test_images_original))
            self.evaluator.plot_sample_predictions(
                test_images_original[:sample_size], 
                y_test[:sample_size], 
                predictions[:sample_size]
            )
        
        return predictions, accuracy
    
    def predict_single_image(self, image_path):
        """Predict emotion for a single image"""
        # Load image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect and preprocess face
        face, bbox = self.face_detector.detect_face(img)
        if face is None:
            return None, "No face detected"
        
        processed = self.face_detector.preprocess_face(face)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            processed.reshape(1, processed.shape[0], processed.shape[1])
        )
        
        # Predict
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        emotion = self.classifier.emotion_map[prediction]
        
        return {
            'emotion': emotion,
            'probabilities': {self.classifier.emotion_map[i]: prob 
                            for i, prob in enumerate(probabilities)},
            'bbox': bbox
        }, None


# ============================================================================
# PART 7: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print(" "*15 + "FACIAL MOOD DETECTION SYSTEM")
    print(" "*10 + "Using OpenCV and CK+48 Dataset")
    print("="*70 + "\n")
    
    # ========================================================================
    # CONFIGURATION - MODIFY THESE SETTINGS
    # ========================================================================
    
    # Dataset path - CHANGE THIS TO YOUR DATASET LOCATION
    DATASET_PATH = './ck+48'  # <<< UPDATE THIS PATH
    
    # Set to False when you have the real CK+48 dataset
    USE_SYNTHETIC_DATA = False  # <<< SET TO False FOR REAL DATA
    
    # Classifier type: 'svm' (recommended), 'knn', or 'ann'
    CLASSIFIER_TYPE = 'svm'
    
    # Feature extraction method: 'eigenface' (recommended), 'hog', or 'flatten'
    FEATURE_METHOD = 'eigenface'
    
    # ========================================================================
    
    try:
        # Step 1: Load Data
        print("[Step 1/7] Loading dataset...")
        print("-" * 70)
        
        data_loader = CKPlusDataLoader(DATASET_PATH)
        
        if USE_SYNTHETIC_DATA:
            print("[WARNING] Using synthetic data for demonstration...")
            images, labels = data_loader.create_synthetic_dataset(num_samples=500)
            X_train, X_test, y_train, y_test = train_test_split(
                images, labels, test_size=0.2, random_state=42, stratify=labels
            )
        else:
            # Load real CK+48 dataset
            result = data_loader.load_ck48_dataset()
            
            if len(result) == 4:
                # Dataset has train/test split
                print("[OK] Using pre-split train/test data")
                X_train, y_train, X_test, y_test = result
            else:
                # No split, create our own
                print("[OK] Creating train/test split (80/20)...")
                images, labels = result
                X_train, X_test, y_train, y_test = train_test_split(
                    images, labels, test_size=0.2, random_state=42, 
                    stratify=labels
                )
        
        # Display dataset statistics
        print(f"\n[STATS] Dataset Statistics:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Image shape: {X_train[0].shape}")
        print(f"   Number of classes: {len(np.unique(y_train))}")
        
        # Display class distribution
        print(f"\n[DISTRIBUTION] Class Distribution (Training):")
        emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 
                       3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            percentage = (count / len(y_train)) * 100
            print(f"   {emotion_map[label]:<10}: {count:>4} images ({percentage:>5.1f}%)")
        
        # Step 2: Initialize Pipeline
        print(f"\n[Step 2/7] Initializing pipeline...")
        print("-" * 70)
        print(f"Classifier: {CLASSIFIER_TYPE.upper()}")
        print(f"Feature extraction: {FEATURE_METHOD}")
        
        pipeline = MoodDetectionPipeline(
            classifier_type=CLASSIFIER_TYPE,
            feature_method=FEATURE_METHOD
        )
        
        # Step 3: Preprocess Images
        print(f"\n[Step 3/7] Preprocessing images...")
        print("-" * 70)
        # For CK+48, images are already cropped faces, so just normalize
        X_train_processed = X_train.astype(np.float32) / 255.0
        X_test_processed = X_test.astype(np.float32) / 255.0
        print("[OK] Images normalized to [0, 1] range")
        
        # Step 4: Visualize Data Distribution
        print(f"\n[Step 4/7] Visualizing data distribution...")
        print("-" * 70)
        pipeline.evaluator.plot_class_distribution(
            y_train, 
            title="Training Set - Class Distribution"
        )
        
        # Step 5: Train Model
        print(f"\n[Step 5/7] Training model...")
        print("-" * 70)
        X_train_features = pipeline.train_pipeline(X_train_processed, y_train)
        
        # Step 6: Evaluate Model
        print(f"\n[Step 6/7] Evaluating model...")
        print("-" * 70)
        predictions, accuracy = pipeline.evaluate_pipeline(
            X_test_processed, y_test, X_test
        )
        
        # Display per-class accuracy
        print("\n[STATS] Per-Class Performance:")
        for label in sorted(np.unique(y_test)):
            mask = y_test == label
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], predictions[mask])
                print(f"   {emotion_map[label]:<10}: {class_acc*100:>5.2f}%")
        
        # Step 7: Save Model
        print(f"\n[Step 7/7] Saving model...")
        print("-" * 70)
        model_path = f'mood_detection_{CLASSIFIER_TYPE}_ck48.xml'
        pipeline.classifier.save_model(model_path)
        
        # Save feature extractor (if using PCA)
        if pipeline.feature_extractor.pca is not None:
            pca_path = 'feature_extractor_pca.pkl'
            with open(pca_path, 'wb') as f:
                pickle.dump(pipeline.feature_extractor.pca, f)
            print(f"Feature extractor saved to: {pca_path}")
        
        # Final Summary
        print("\n" + "="*70)
        print(" "*20 + "[OK] TRAINING COMPLETED!")
        print("="*70)
        print(f"\n[RESULT] Model saved as: {model_path}")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {e}")
        print(f"Please ensure your dataset is in: {DATASET_PATH}")
        print("Or set USE_SYNTHETIC_DATA = True to use demonstration data")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()