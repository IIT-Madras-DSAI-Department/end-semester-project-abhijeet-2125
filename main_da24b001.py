# DA24B001
import numpy as np
import pandas as pd
import time, os
from collections import Counter, defaultdict

# --- Utility functions ---
def one_hot(y, n_classes=None):
    y = np.array(y, dtype=int)
    if n_classes is None:
        n_classes = np.max(y) + 1
    oh = np.zeros((y.shape[0], n_classes), dtype=float)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot_stack_features(label_array, n_classes=10):
    # This check is added for safety
    if label_array.ndim == 1:
        label_array = label_array.reshape(-1, 1)
        
    n_samples, n_models = label_array.shape
    ohe_features = np.zeros((n_samples, n_models * n_classes))
    for i in range(n_models):
        labels = label_array[:, i]
        ohe = one_hot(labels, n_classes=n_classes)
        ohe_features[:, i*n_classes : (i+1)*n_classes] = ohe
    return ohe_features

# --- Model classes ---

# 1. Multinomial Logistic Regression (Base Model + Meta-Model)
class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.05, n_epochs=50, batch_size=128, reg_lambda=0.001, random_state=42, early_stopping_rounds=5):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.W = None
    def _init(self, n_features, n_classes):
        rng = np.random.RandomState(self.random_state)
        self.W = 0.01 * rng.randn(n_features, n_classes)
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X); y = np.asarray(y, dtype=int)
        n_samples, n_features = X.shape
        n_classes = np.max(y) + 1
        if n_features == 0:
            print("Warning: 0 features passed to LR.fit. Check OHE.")
            n_features = X.shape[1]
        self._init(n_features, n_classes)
        yoh = one_hot(y, n_classes)
        for epoch in range(self.n_epochs):
            perm = np.random.permutation(n_samples)
            Xs = X[perm]; ys = yoh[perm]
            for i in range(0, n_samples, self.batch_size):
                xb = Xs[i:i+self.batch_size]; yb = ys[i:i+self.batch_size]
                if xb.size==0: continue
                probs = softmax(xb @ self.W)
                grad = xb.T @ (probs - yb) / xb.shape[0]
                grad += self.reg_lambda * self.W
                self.W -= self.learning_rate * grad
        return self
    def predict_proba(self, X):
        return softmax(np.asarray(X) @ self.W)
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# 2. Powerful XGBoost (Binary Engine)
class XGBDecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class XGBoostClassifier:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_pred = None
        self.n_sub_features = None
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    def _compute_initial_prediction(self, y):
        p = np.mean(y)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log(p / (1 - p))
    def _compute_gradients_hessians(self, y_true, y_pred_proba):
        gradients = y_pred_proba - y_true
        hessians = y_pred_proba * (1 - y_pred_proba)
        return gradients, hessians
    def _calculate_leaf_value(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.reg_lambda)
    def _calculate_gain(self, g, h, left_idx, right_idx):
        g_left, h_left = g[left_idx], h[left_idx]
        g_right, h_right = g[right_idx], h[right_idx]
        G_left, H_left = np.sum(g_left), np.sum(h_left)
        G_right, H_right = np.sum(g_right), np.sum(h_right)
        G_parent, H_parent = np.sum(g), np.sum(h)
        gain_left = (G_left**2) / (H_left + self.reg_lambda) if (H_left + self.reg_lambda) != 0 else 0.0
        gain_right = (G_right**2) / (H_right + self.reg_lambda) if (H_right + self.reg_lambda) != 0 else 0.0
        gain_parent = (G_parent**2) / (H_parent + self.reg_lambda) if (H_parent + self.reg_lambda) != 0 else 0.0
        gain = 0.5 * (gain_left + gain_right - gain_parent)
        return gain
    def _find_best_split(self, X, g, h):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        n_samples, n_features = X.shape
        if self.n_sub_features is None:
             self.n_sub_features = int(np.sqrt(n_features)) + 1
        feature_indices = np.random.choice(n_features, self.n_sub_features, replace=False)
        for feat_idx in feature_indices:
            if X.shape[0] > 1 and np.all(X[:, feat_idx] == X[0, feat_idx]):
                continue
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) > 10:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 10))
            for thresh in thresholds:
                left_idx = X[:, feat_idx] <= thresh
                right_idx = X[:, feat_idx] > thresh
                if len(g[left_idx]) == 0 or len(g[right_idx]) == 0:
                    continue
                gain = self._calculate_gain(g, h, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain
    def _build_tree(self, X, g, h, depth):
        n_samples = len(g)
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(g, h))
        best_feat, best_thresh, best_gain = self._find_best_split(X, g, h)
        if best_gain <= 0 or best_feat is None:
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(g, h))
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = X[:, best_feat] > best_thresh
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(g, h))
        left = self._build_tree(X[left_idx], g[left_idx], h[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], g[right_idx], h[right_idx], depth + 1)
        return XGBDecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_sub_features = int(np.sqrt(n_features)) + 1 
        self.trees = []
        self.base_pred = self._compute_initial_prediction(y)
        current_predictions_raw = np.full(n_samples, self.base_pred)
        for i in range(self.n_estimators):
            y_pred_proba = self._sigmoid(current_predictions_raw)
            g, h = self._compute_gradients_hessians(y, y_pred_proba)
            tree = self._build_tree(X, g, h, depth=0)
            self.trees.append(tree)
            update = self.learning_rate * self._predict_tree(X, tree)
            current_predictions_raw = current_predictions_raw + update
        return self
    def _traverse_tree(self, inputs, node):
        if node.is_leaf_node():
            return node.value
        if node.feature_index is None:
             return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._traverse_tree(inputs, node.left)
        else:
            return self._traverse_tree(inputs, node.right)
    def _predict_tree(self, X, tree):
        return np.array([self._traverse_tree(inputs, tree) for inputs in X])
    def predict_proba(self, X):
        X = np.asarray(X)
        raw_predictions = np.full(X.shape[0], self.base_pred)
        for tree in self.trees:
            raw_predictions = raw_predictions + self.learning_rate * self._predict_tree(X, tree)
        return self._sigmoid(raw_predictions)
    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)

# 3. Multiclass XGBoost Wrapper
class MulticlassXGBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0):
        print("Initializing MulticlassXGBoost (OvR) with max_depth=" + str(max_depth))
        self.n_classes = 10 # Hardcoded for MNIST
        self.models_per_class = []
        self.base_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'reg_lambda': reg_lambda
        }
    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        for k in range(self.n_classes):
            print("--- Training model for class " + str(k) + " ---")
            y_binary = (y == k).astype(int)
            model_k = XGBoostClassifier(**self.base_params)
            model_k.fit(X, y_binary)
            self.models_per_class.append(model_k)
        print("Multiclass XGBoost training completed")
        return self
    def predict(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        all_probas = np.zeros((n_samples, self.n_classes))
        for k in range(self.n_classes):
            probas_k = self.models_per_class[k].predict_proba(X)
            all_probas[:, k] = probas_k
        return np.argmax(all_probas, axis=1)

# --- Preprocessing Function ---
def load_and_preprocess(train_file_path, test_file_path):
    print("Loading training data for preprocessing...")
    try:
        train_df = pd.read_csv(train_file_path)
    except FileNotFoundError:
        print("Fatal Error: Training file not found at " + str(train_file_path))
        return None, None, None, None, None
        
    X_train_raw = train_df.drop('label', axis=1).values.astype(float) / 255.0
    y_train = train_df['label'].values.astype(int)

    print("Loading test data...")
    try:
        test_df = pd.read_csv(test_file_path)
    except FileNotFoundError:
        print("Fatal Error: Test file not found at " + str(test_file_path))
        return None, None, None, None, None
        
    if 'label' in test_df.columns:
        X_test_raw = test_df.drop('label', axis=1).values.astype(float)/255.0
    else:
        X_test_raw = test_df.values.astype(float)/255.0

    print("Applying Variance Threshold...")
    variance_threshold = 1e-5
    feature_variances = np.var(X_train_raw, axis=0)
    mask = feature_variances > variance_threshold
    
    X_train_filt = X_train_raw[:, mask]
    X_test_filt = X_test_raw[:, mask]

    print("Applying Mean Centering...")
    X_mean = np.mean(X_train_filt, axis=0)
    X_train_centered = X_train_filt - X_mean
    X_test_centered = X_test_filt - X_mean
    
    clip_limit = 3 * np.std(X_train_centered)
    X_train_centered = np.clip(X_train_centered, -clip_limit, clip_limit)
    X_test_centered = np.clip(X_test_centered, -clip_limit, clip_limit)

    print("Fitting PCA on training data...")
    U, S, VT = np.linalg.svd(X_train_centered, full_matrices=False)
    explained_variance = (S**2) / (X_train_centered.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    cumulative = np.cumsum(explained_variance_ratio)
    
    n_components = int(np.searchsorted(cumulative, 0.92) + 1)
    print("PCA retaining 92% variance -> " + str(n_components) + " components.")
    components = VT[:n_components].T

    print("Transforming train and test data with PCA...")
    X_train_pca = X_train_centered.dot(components)
    X_test_pca = X_test_centered.dot(components)
    
    return X_train_pca, y_train, X_test_pca

# --- Main Execution ---
def main():
    TRAIN_CSV = 'MNIST_train.csv'
    TEST_CSV = 'MNIST_test.csv'
    OUT_PRED = 'predictions.csv'
    start_total = time.time()

    X_train_pca, y_train, X_test_pca = load_and_preprocess(TRAIN_CSV, TEST_CSV)
    if X_train_pca is None:
        return
    
    # This base_list MUST be inside the main() function
    base_list = [
        ('lr', MultinomialLogisticRegression, {'learning_rate': 0.1, 'n_epochs': 200, 'batch_size': 64, 'reg_lambda': 0.0001}),
        ('xgb', MulticlassXGBoost, {'n_estimators': 60, 'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 10, 'reg_lambda': 1.0})
    ]

    fitted = []
    for name, EstClass, kwargs in base_list:
        print("Training final base model: " + str(name) + " with " + str(kwargs))
        est = EstClass(**kwargs)
        est.fit(X_train_pca, y_train)
        fitted.append((name, est))
    
    meta_X_labels = [est.predict(X_train_pca).reshape(-1,1) for _, est in fitted]
    meta_X = np.hstack(meta_X_labels)

    print("Original meta-feature shape: " + str(meta_X.shape))
    meta_X_ohe = one_hot_stack_features(meta_X)
    print("New OHE meta-feature shape: " + str(meta_X_ohe.shape))

    meta_clf = MultinomialLogisticRegression(learning_rate=0.1, n_epochs=80, batch_size=128, reg_lambda=0.001)
    meta_clf.fit(meta_X_ohe, y_train) # Fit on OHE features

    meta_test_labels = [est.predict(X_test_pca).reshape(-1,1) for _, est in fitted]
    meta_test_X = np.hstack(meta_test_labels)
    meta_test_X_ohe = one_hot_stack_features(meta_test_X)

    preds = meta_clf.predict(meta_test_X_ohe)
    out_df = pd.DataFrame({'Id': np.arange(len(preds)), 'Label': preds})
    out_df.to_csv(OUT_PRED, index=False)

    end_total = time.time()
    print("\nSaved predictions to " + str(OUT_PRED))
    print("Total script runtime: " + str(round(end_total - start_total, 2)) + " seconds.")

if __name__ == "__main__":
    main()