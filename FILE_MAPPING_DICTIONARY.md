# THUOC PROJECT - STRUCTURED REFERENCE DICTIONARY

## FILE FUNCTION MAPPING (JSON Format)

```json
{
  "root_files": {
    "run_all.py": {
      "function": "Main entry point - orchestrate all training, evaluation, and visualization",
      "key_operations": [
        "Discover data directory (data_aligned or data)",
        "Train 3 models (resnet50, efficientnet_b0, vit_b_16) sequentially or individually",
        "Run evaluation on test set for each model",
        "Generate comparison metrics and visualizations",
        "Support CLI args: --model, --compare-only, --device, --data-dir"
      ],
      "dependencies": ["optimal_configs.py", "src.pipeline", "src.train"]
    },
    "train_cli.py": {
      "function": "CLI wrapper for training with 3 modes: all (pipeline), single (1 model), optimize (hyperparameter tuning)",
      "key_operations": [
        "Parse command line arguments for training mode",
        "Invoke train.train() or pipeline.main() or review_terminal interactive tuning",
        "Support custom configurations for each mode"
      ],
      "modes": {
        "all": "Run full pipeline (train 3 models + eval)",
        "single": "Train single model with custom hyperparams",
        "optimize": "Interactive tuning loop with real-time feedback"
      }
    },
    "run_gui.py": {
      "function": "Launcher for Tkinter GUI desktop application",
      "launches": "src.gui_tk.PillClassifierApp",
      "features": [
        "Compare two pill images (sample vs. query)",
        "Display similarity scores (overall, color, size, texture)",
        "Show medicine metadata from CSV",
        "Model selection dropdown (resnet50, efficientnet_b0, vit_b_16)",
        "Device selection (cuda or cpu)"
      ]
    },
    "review_terminal.py": {
      "function": "Interactive real-time training review and hyperparameter suggestion system",
      "key_operations": [
        "Monitor training curves as epochs progress",
        "Detect overfitting patterns (large train-val gap)",
        "Suggest hyperparameter adjustments (lr, weight_decay, mixup_alpha, etc.)",
        "Support manual intervention during training",
        "Save tuning history for analysis"
      ]
    },
    "optimal_configs.py": {
      "function": "Pre-optimized hyperparameter configurations for each model",
      "exports": {
        "OPTIMAL_CONFIGS": {
          "description": "Best configs tuned for small datasets to minimize overfitting",
          "keys": ["resnet50", "efficientnet_b0", "vit_b_16"],
          "config_params": [
            "lr (learning rate)",
            "weight_decay (L2 regularization)",
            "label_smoothing (CrossEntropy smoothing)",
            "mixup_alpha (data augmentation strength)",
            "epochs",
            "batch_size",
            "early_stop_patience",
            "max_train_val_gap (divergence guard)",
            "freeze_backbone_epochs"
          ]
        },
        "TUNING_CANDIDATES": {
          "description": "Alternative hyperparameter sets for experimentation",
          "keys": ["resnet50", "efficientnet_b0", "vit_b_16"],
          "per_model": "3 candidate configurations for grid search"
        }
      }
    },
    "requirements.txt": {
      "function": "Python package dependencies with versions",
      "format": "pip-compatible format (package_name>=version)",
      "packages": {
        "torch": ">=2.0.0 - PyTorch deep learning framework",
        "torchvision": ">=0.15.0 - Pretrained models (ResNet50, EfficientNet, ViT)",
        "numpy": ">=1.24.0 - Numerical operations",
        "Pillow": ">=10.0.0 - Image I/O and processing",
        "scikit-learn": ">=1.3.0 - Metrics and validation",
        "tqdm": ">=4.66.0 - Progress bars",
        "matplotlib": ">=3.7.0 - Visualization and plotting",
        "pandas": ">=2.0.0 - CSV I/O and dataframe operations"
      }
    },
    "README.md": {
      "function": "User-facing quick start guide (Vietnamese)",
      "contents": [
        "Project overview",
        "Quick setup (5 min)",
        "Important commands reference table",
        "Expected outputs",
        "Google Colab usage",
        "Current benchmark results"
      ]
    }
  },
  
  "src_files": {
    "train.py": {
      "function": "Core training engine - implements main training loop, loss computation, and model optimization",
      "main_functions": {
        "parse_args()": "Parse command line arguments (lr, epochs, batch_size, etc.)",
        "create_dataloaders()": {
          "description": "Build train/val/test DataLoaders with stratified holdout if val too small",
          "returns": ["train_loader", "val_loader", "train_metric_loader"]
        },
        "_mixup_batch()": {
          "description": "Apply mixup augmentation to image batch",
          "parameters": ["images", "labels", "alpha", "device"],
          "returns": ["mixed_images", "labels_a", "labels_b", "lambda"]
        },
        "evaluate()": {
          "description": "Compute accuracy and loss on validation/test set",
          "parameters": ["model", "loader", "device", "criterion"],
          "returns": ["accuracy", "loss"]
        },
        "_plot_training_curves()": "Save training/validation curves as PNG with EMA smoothing",
        "train()": {
          "description": "Main training loop for single model",
          "stages": [
            "Stage 1: Freeze backbone (first N epochs)",
            "Stage 2: Unfreeze backbone (remaining epochs)",
            "Warmup: Linear LR increase (first 3 epochs)",
            "Per-epoch: forward → mixup loss → backward → optimizer step",
            "Evaluation: compute train/val metrics",
            "Scheduling: reduce LR on plateau",
            "Early stopping: patience or train-val divergence guard"
          ],
          "outputs": ["checkpoint.pt", "metrics.json", "history.json", "curves.png"]
        }
      },
      "key_hyperparams": {
        "optimizer": "AdamW (lr, weight_decay)",
        "loss_fn": "CrossEntropyLoss (label_smoothing)",
        "scheduler": "ReduceLROnPlateau (factor=0.7, patience=2)",
        "regularization": ["weight_decay", "label_smoothing", "mixup_alpha", "grad_clip_norm"],
        "stopping": ["early_stop_patience", "max_train_val_gap"]
      }
    },
    
    "models.py": {
      "function": "Model creation and checkpoint loading - defines 3 architectures and class replacement",
      "main_functions": {
        "create_model()": {
          "description": "Build classification model with pretrained backbone",
          "parameters": [
            "model_name: 'resnet50'|'efficientnet_b0'|'vit_b_16'",
            "num_classes: int",
            "pretrained: bool",
            "fallback_to_random: bool (if pretrained download fails)"
          ],
          "process": [
            "Download pretrained weights from torchvision",
            "Replace final classification layer with num_classes output",
            "Fallback to random init if network unavailable"
          ],
          "returns": ["model (nn.Module)", "feature_dim (int)"]
        },
        "load_checkpoint()": {
          "description": "Load model from saved checkpoint",
          "parameters": ["model_name", "num_classes", "checkpoint_path", "map_location"],
          "process": [
            "Load .pt file (handles both dict and raw state_dict)",
            "Infer num_classes from checkpoint if available",
            "Create model and load state_dict"
          ],
          "returns": ["model (nn.Module)"]
        },
        "load_checkpoint_class_to_idx()": {
          "description": "Extract class mapping from checkpoint metadata",
          "parameters": ["checkpoint_path", "map_location"],
          "returns": ["Dict[str, int] | None"]
        }
      },
      "architectures": {
        "resnet50": {
          "source": "torchvision.models.resnet50",
          "weights": "ImageNet1K_V2 (best variant)",
          "backbone_layers": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
          "feature_dim": 2048,
          "classifier_replacement": "model.fc = Linear(2048, num_classes)"
        },
        "efficientnet_b0": {
          "source": "torchvision.models.efficientnet_b0",
          "weights": "ImageNet1K_V1",
          "backbone_module": "features (MBConv blocks)",
          "feature_dim": 1280,
          "classifier_replacement": "model.classifier[-1] = Linear(1280, num_classes)"
        },
        "vit_b_16": {
          "source": "torchvision.models.vit_b_16",
          "weights": "ImageNet1K_V1",
          "backbone_module": "encoder (transformer blocks)",
          "feature_dim": 768,
          "classifier_replacement": "model.heads.head = Linear(768, num_classes)"
        }
      }
    },
    
    "features.py": {
      "function": "Data loading, augmentation pipeline, and dataset class",
      "main_classes": {
        "PillImageDataset(Dataset)": {
          "description": "PyTorch Dataset for pill image classification",
          "__init__": "Initialize with root directory path and transforms",
          "_find_classes()": "Scan directory structure and build class_to_idx mapping",
          "__len__()": "Return total number of images",
          "__getitem__()": {
            "process": [
              "Load image from path",
              "Convert to RGB",
              "Apply focus_on_object (center crop 85%)",
              "Apply transforms (resize, augment, normalize)",
              "Return (tensor, label, path)"
            ]
          }
        }
      },
      "main_functions": {
        "build_transforms()": {
          "description": "Create augmentation pipeline",
          "training_transforms": [
            "Resize(224, 224)",
            "ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08)",
            "RandomHorizontalFlip()",
            "RandomRotation(5°)",
            "ToTensor",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
          ],
          "eval_transforms": [
            "Resize(224, 224)",
            "ToTensor",
            "Normalize(same as training)"
          ]
        },
        "pil_loader()": "Load image file and convert to RGB",
        "focus_on_object()": {
          "description": "Center crop image to focus on pill (remove background)",
          "parameter": "scale=0.85 (keep 85% of min dimension)",
          "process": "Center crop with side = min(w, h) * scale"
        },
        "compute_image_statistics()": {
          "description": "Extract features for color/texture scoring",
          "returns": {
            "mean_r": "Average red channel",
            "mean_g": "Average green channel",
            "mean_b": "Average blue channel",
            "height": "Image height (after focus)",
            "width": "Image width",
            "aspect_ratio": "width / height"
          }
        }
      },
      "constants": {
        "IMG_SIZE": "224",
        "VALID_EXTS": ["jpg", "jpeg", "png"],
        "IMAGENET_MEAN": "[0.485, 0.456, 0.406]",
        "IMAGENET_STD": "[0.229, 0.224, 0.225]"
      }
    },
    
    "pipeline.py": {
      "function": "High-level orchestration of training, evaluation, and reporting for all 3 models",
      "main_functions": {
        "discover_data_dir()": "Auto-detect data root (prefer data_aligned, fallback data)",
        "_evaluate_single_model()": "Run inference on test set and compute metrics (Acc, F1, confusion matrix)",
        "main()": {
          "stages": [
            "Stage 1: Data discovery and validation",
            "Stage 2: Train 3 models (sequentially)",
            "Stage 3: Evaluate each model on test set",
            "Stage 4: Generate comparison metrics",
            "Stage 5: Create visualizations (confmat, comparison bars)"
          ],
          "outputs": ["CSV report", "PNG charts", "JSON summaries"]
        }
      },
      "dataclass_outputs": {
        "ModelEvalResult": {
          "fields": ["model", "accuracy", "macro_f1", "num_samples", "checkpoint"]
        },
        "PipelineSummary": {
          "fields": [
            "started_at", "finished_at", "elapsed_seconds",
            "data_dir", "test_dir", "models_dir", "report_dir",
            "trained_models", "best_model", "ensemble_model_name"
          ]
        }
      }
    },
    
    "inference.py": {
      "function": "Model inference and image comparison - end-to-end prediction and similarity scoring",
      "main_classes": {
        "ComparisonResult": {
          "fields": [
            "predicted_class: str",
            "similarity_score: float (0-1, higher = more similar)",
            "color_score: float",
            "size_score: float",
            "texture_score: float",
            "num_true_features: int",
            "is_true: bool",
            "details: Dict"
          ]
        }
      },
      "main_functions": {
        "_get_eval_transform()": "Get cached eval transforms (lazy initialization)",
        "_get_or_load_model()": "Load model from checkpoint with caching to avoid reloads",
        "compare_pill_images()": {
          "description": "Compare sample and query image with specified model",
          "inputs": ["sample_img_path", "query_img_path", "model_name", "class_to_idx"],
          "process": [
            "Load and preprocess both images",
            "Extract features from backbone",
            "Compute feature-level similarity (cosine distance)",
            "Compute color, size, texture similarity",
            "Generate prediction with thresholds",
            "Return ComparisonResult"
          ]
        },
        "compare_pill_images_auto()": "Ensemble version - try all 3 models and vote on prediction",
        "_resolve_device()": "Auto-fallback from CUDA to CPU if GPU unavailable"
      },
      "caching": {
        "_MODEL_CACHE": "Global dict keyed by (model_name, checkpoint_path, device)",
        "purpose": "Avoid reloading same model multiple times in GUI session"
      }
    },
    
    "evaluate_report.py": {
      "function": "Compute evaluation metrics and export formatted reports",
      "main_functions": {
        "_evaluate_one_model()": {
          "description": "Run inference on test set and compute accuracy + macro-F1",
          "parameters": ["model_name", "checkpoint_path", "loader", "dataset_class_to_idx", "device"],
          "returns": {
            "accuracy": "Float (0-1)",
            "macro_f1": "Float (0-1)",
            "num_samples": "Int"
          }
        },
        "main()": {
          "process": [
            "Load test data",
            "Evaluate each model from checkpoint",
            "Export CSV results (accuracy, F1)",
            "Generate comparison bar chart (PNG)"
          ]
        }
      }
    },
    
    "build_epillid_data.py": {
      "function": "Integrate ePillID source data into THUOC format (copy/link files into train/val/test splits)",
      "main_functions": {
        "load_epillid_splits()": "Read CSV splits from ePillID source",
        "_build_class_name()": "Generate class folder name from pill metadata + side (front/back)",
        "_file_op_copy()": {
          "description": "Flexible file operation (copy, hardlink, or symlink)",
          "modes": ["copy", "hardlink", "symlink"],
          "fallback": "If hardlink/symlink fails, fallback to copy"
        },
        "main()": {
          "output": "Create data_aligned/ directory with train/val/test splits"
        }
      }
    },
    
    "metadata.py": {
      "function": "Parse and match medicine metadata from CSV to class names",
      "main_classes": {
        "MedicineMetadataRecord": {
          "fields": [
            "medicine_name", "composition", "dosage_form", "weight",
            "color", "shape", "active_group", "disease_vi"
          ]
        },
        "MedicineMetadataIndex": {
          "description": "Build in-memory index for fast metadata lookup",
          "from_csv()": "Load records from CSV and build token index",
          "best_match()": {
            "description": "Find best CSV row matching a class folder name using token overlap",
            "algorithm": "Dice score with token intersection",
            "threshold": "Minimum score 0.2 to avoid weak matches"
          }
        }
      },
      "main_functions": {
        "normalize_text()": "Normalize text for robust matching (remove accents, strip punctuation)",
        "_tokenize()": "Split normalized text into token set"
      }
    },
    
    "self_learning.py": {
      "function": "Capture user feedback and build datasets of hard/wrong cases for fine-tuning",
      "main_classes": {
        "FeedbackRecord": {
          "fields": [
            "sample_image_path", "query_image_path",
            "predicted_class", "is_true_system", "is_true_user",
            "model_name", "similarity_score", "color/size/texture_scores"
          ]
        }
      },
      "main_functions": {
        "log_feedback()": "Append user feedback record to JSONL log file",
        "load_feedback()": "Read all feedback records from JSONL",
        "build_hard_example_lists()": {
          "description": "Extract subsets for fine-tuning",
          "outputs": {
            "system_wrong": "Cases where model disagrees with user label",
            "hard_cases": "High confidence but incorrect predictions"
          }
        }
      }
    },
    
    "gui_tk.py": {
      "function": "Desktop GUI application for interactive pill comparison and classification",
      "main_class": {
        "PillClassifierApp(tk.Tk)": {
          "features": [
            "Load sample pills from demo_images/",
            "Compare sample vs. user-uploaded query image",
            "Display similarity scores (overall, color, size, texture)",
            "Show medicine metadata from CSV",
            "Model selector (resnet50, efficientnet_b0, vit_b_16)",
            "Device selector (cuda, cpu)",
            "Threshold sliders for each score type"
          ],
          "key_methods": [
            "_build_class_mapping()",
            "_load_demo_entries()",
            "_compare_pills_thread()",
            "_on_select_image()",
            "_update_results_table()"
          ]
        }
      }
    }
  },

  "data_pipeline_flow": {
    "data_discovery": "pipeline.py::discover_data_dir() → find data_aligned or data/",
    "class_mapping": "PillImageDataset._find_classes() → {classname: idx}",
    "data_loading": "PillImageDataset.__init__() → scan files in train/val/test",
    "augmentation": "features.py::build_transforms() → ColorJitter, Flip, Rotate, normalize",
    "batching": "train.py::create_dataloaders() → train/val/test DataLoaders",
    "stratified_holdout": "If val too small, split part of train into val (stratified by class)"
  },

  "training_flow": {
    "1_setup": "Seed, dataloaders, model download, optimizer, scheduler init",
    "2_per_epoch": {
      "2a_backbone_freeze": "If epoch <= freeze_epochs: freeze backbone",
      "2b_warmup": "If epoch <= 3: linear warmup to base_lr",
      "2c_batch_loop": "For each batch: (1) mixup, (2) forward, (3) loss, (4) backward, (5) clip grad, (6) optimizer step",
      "2d_eval": "Compute train_acc (clean) and val_acc",
      "2e_schedule": "Reduce LR if val_loss plateau",
      "2f_early_stop": "Check patience or train-val gap divergence"
    },
    "3_checkpoint": "If val_acc improves: save model + class_to_idx + metrics",
    "4_output": "Save history JSON, training curves PNG"
  },

  "inference_flow": {
    "1_load": "Load model checkpoint (with class_to_idx metadata)",
    "2_preprocess": "PIL loader → focus_on_object → resize → normalize",
    "3_forward": "model(image) → logits",
    "4_predict": "argmax(logits) → class name",
    "5_score": "softmax for confidence",
    "6_optional_ensemble": "Repeat for 3 models, vote on final class"
  },

  "key_concepts": {
    "pretrained_weights": "Download ImageNet weights, replace only final layer, train with frozen backbone first",
    "mixup": "Blend pairs of images at random ratio λ~Beta(α,α), blend losses similarly",
    "label_smoothing": "Soften target distribution: y_true → (1-ε)*y + ε/K for K classes",
    "early_stopping": "Stop if validation accuracy doesn't improve for N epochs OR train-val gap too large",
    "gradient_clipping": "Clip gradient norm to prevent exploding gradients during backprop",
    "checkpoint_class_mapping": "Save class_to_idx in checkpoint to avoid mismatch when loading for inference",
    "stratified_holdout": "If test set too small, use part of train as validation while preserving class balance"
  }
}
```


## QUICKSTART COMMAND REFERENCE

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 3 models (full pipeline)
python run_all.py

# Train single model
python run_all.py --model resnet50

# Only evaluate existing models (no training)
python run_all.py --compare-only

# Use CPU instead of GPU
python run_all.py --device cpu

# Custom data directory
python run_all.py --data-dir data_aligned

# Launch GUI
python run_gui.py

# CLI mode: full pipeline
python train_cli.py --mode all --data-dir data_aligned

# CLI mode: single model training
python train_cli.py --mode single --model efficientnet_b0 --epochs 20

# CLI mode: interactive tuning
python train_cli.py --mode optimize --rounds 5
```


## FILE SIZE & DESCRIPTION

| File | Lines | Purpose |
|------|-------|---------|
| train.py | ~700 | Training loop, forward/backward, evaluation, checkpointing |
| models.py | ~150 | Model creation and checkpoint loading |
| features.py | ~180 | Dataset class, augmentation transforms |
| pipeline.py | ~300 | Orchestration, evaluation, report generation |
| inference.py | ~250 | Model inference, image comparison, feature extraction |
| gui_tk.py | ~400+ | Tkinter GUI desktop app |
| evaluate_report.py | ~100 | Evaluation metrics and CSV export |
| build_epillid_data.py | ~250 | Data integration from ePillID source |
| metadata.py | ~150 | CSV metadata parsing and matching |
| self_learning.py | ~100 | Feedback logging for hard examples |
| run_all.py | ~150 | Main entry point, CLI orchestration |
| train_cli.py | ~100 | CLI argument parsing and dispatch |
| optimal_configs.py | ~50 | Hyperparameter configurations |

---

**Note:** All paths use forward slashes internally (os.path.join handles conversion on Windows)
