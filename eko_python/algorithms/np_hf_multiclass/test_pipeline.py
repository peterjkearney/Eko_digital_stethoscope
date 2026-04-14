"""
test_pipeline.py

Tests for dataset.py, model.py, and train.py.

All tests are self-contained — they create synthetic PNG spectrograms and
manifests in a tmp_path fixture so no real data is required.

Run with:  pytest test_pipeline.py -v
"""

import sys
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (
    HFLungDataset,
    extract_patient_date,
    get_dataloaders,
    get_train_groups,
    LABEL_COLS,
)
from model import (
    LungSoundClassifier,
    MultiLabelBCELoss,
    build_model,
    save_checkpoint,
    load_checkpoint,
)
from train import evaluate, train_one_epoch

H, W = 224, 224


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_png(path: Path, h: int = H, w: int = W) -> None:
    arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    Image.fromarray(arr).save(str(path))


def make_manifest(
    tmp_path: Path,
    n_train: int = 6,
    n_test:  int = 4,
    n_aug_per_window: int = 0,
) -> tuple[Path, Path]:
    """
    Create synthetic PNGs and a manifest CSV.
    recording_ids use the steth format so extract_patient_date works.
    Train windows are spread across two patient-dates for GroupKFold.
    Returns (manifest_path, spectrograms_dir).
    """
    spec_dir = tmp_path / 'spectrograms'
    (spec_dir / 'train').mkdir(parents=True)
    (spec_dir / 'test').mkdir(parents=True)

    rows = []
    dates = ['20180101', '20180102']

    for i in range(n_train):
        date   = dates[i % len(dates)]
        rec_id = f'steth_{date}_09_00_0{i}'
        stem   = f'train_{i:04d}'
        png    = spec_dir / 'train' / f'{stem}.png'
        make_png(png)
        rows.append({
            'recording_id': rec_id, 'device': 'steth', 'split': 'train',
            'window_start': float(i), 'window_end': float(i + 2),
            'das': i % 2, 'cas': (i + 1) % 2,
            'das_overlap': 0.3 * (i % 2), 'cas_overlap': 0.3 * ((i+1) % 2),
            'aug_index': 0, 'spec_path': str(png),
        })
        for aug_i in range(1, n_aug_per_window + 1):
            aug_png = spec_dir / 'train' / f'aug_{i:04d}_{aug_i}.png'
            make_png(aug_png)
            rows.append({
                'recording_id': rec_id, 'device': 'steth', 'split': 'train',
                'window_start': float(i), 'window_end': float(i + 2),
                'das': i % 2, 'cas': (i + 1) % 2,
                'das_overlap': 0.3 * (i % 2), 'cas_overlap': 0.3 * ((i+1) % 2),
                'aug_index': aug_i, 'spec_path': str(aug_png),
            })

    # Ensure both classes present for both labels in the test set
    test_labels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for i in range(n_test):
        rec_id = f'steth_20180201_10_00_0{i}'
        stem   = f'test_{i:04d}'
        png    = spec_dir / 'test' / f'{stem}.png'
        make_png(png)
        das, cas = test_labels[i % len(test_labels)]
        rows.append({
            'recording_id': rec_id, 'device': 'steth', 'split': 'test',
            'window_start': float(i), 'window_end': float(i + 2),
            'das': das, 'cas': cas,
            'das_overlap': 0.3 * das, 'cas_overlap': 0.3 * cas,
            'aug_index': 0, 'spec_path': str(png),
        })

    manifest_path = tmp_path / 'windows_manifest.csv'
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, spec_dir


def tiny_loader(n: int = 32, n_das: int = 10, n_cas: int = 8, batch_size: int = 8):
    """DataLoader backed by random tensors — no disk I/O."""
    images = torch.randn(n, 3, H, W)
    labels = torch.zeros(n, 2)
    labels[:n_das, 0] = 1.0
    labels[:n_cas, 1] = 1.0
    ds = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size)


# ---------------------------------------------------------------------------
# extract_patient_date
# ---------------------------------------------------------------------------

class TestExtractPatientDate:

    def test_steth_format(self):
        assert extract_patient_date('steth_20180814_09_37_11') == '20180814'

    def test_trunc_format(self):
        assert extract_patient_date('trunc_2019-06-03-09-33-45-Tc_1') == '20190603'

    def test_returns_8_chars(self):
        assert len(extract_patient_date('steth_20210301_12_00_00')) == 8

    def test_returns_digits_only(self):
        assert extract_patient_date('steth_20210301_12_00_00').isdigit()


# ---------------------------------------------------------------------------
# HFLungDataset
# ---------------------------------------------------------------------------

class TestHFLungDataset:

    def test_len_train_originals_only(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_test=2, n_aug_per_window=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        assert len(ds) == 4

    def test_len_train_with_augmented(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_test=2, n_aug_per_window=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=True)
        assert len(ds) == 4 + 4 * 2   # 4 orig + 4 windows × 2 aug each

    def test_len_test(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=2, n_test=5)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='test',
                           include_augmented=False)
        assert len(ds) == 5

    def test_getitem_image_shape(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        img, _ = ds[0]
        assert img.shape == (3, H, W)
        assert img.dtype == torch.float32

    def test_getitem_label_shape_and_dtype(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        _, labels = ds[0]
        assert labels.shape == (2,)
        assert labels.dtype == torch.float32

    def test_labels_are_binary(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        for i in range(len(ds)):
            _, y = ds[i]
            assert set(y.tolist()).issubset({0.0, 1.0})

    def test_image_is_normalised(self, tmp_path, monkeypatch):
        """After z-score normalisation, per-sample mean should be ≈ 0."""
        m, s = make_manifest(tmp_path, n_train=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        img, _ = ds[0]
        assert abs(img.mean().item()) < 0.5

    def test_recording_ids_filter(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        manifest = pd.read_csv(m)
        ids = manifest[manifest['split'] == 'train']['recording_id'].unique()[:2].tolist()
        ds = HFLungDataset(manifest_path=m, recording_ids=ids,
                           include_augmented=False)
        assert set(ds.manifest['recording_id'].unique()) == set(ids)

    def test_patient_date_column_present(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        assert 'patient_date' in ds.manifest.columns
        assert ds.manifest['patient_date'].str.len().eq(8).all()

    def test_get_pos_weights_shape(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        w = ds.get_pos_weights()
        assert w.shape == (2,)
        assert (w > 0).all()

    def test_get_pos_weights_reflects_imbalance(self, tmp_path, monkeypatch):
        """All-negative class should get a higher pos_weight than a balanced class."""
        m, s = make_manifest(tmp_path, n_train=6, n_test=0)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        df = pd.read_csv(m)
        df['cas'] = 0   # all negative for CAS
        df.to_csv(m, index=False)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        w = ds.get_pos_weights()
        assert w[1].item() > w[0].item()   # CAS weight > DAS weight

    def test_resample_augmented_one_aug_per_window(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_aug_per_window=3)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=True)
        ds.resample_augmented(seed=0)
        n_orig = (ds.manifest['aug_index'] == 0).sum()
        n_aug  = (ds.manifest['aug_index'] > 0).sum()
        assert n_aug == n_orig   # exactly 1 aug copy per original window

    def test_resample_augmented_preserves_all_originals(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6, n_aug_per_window=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=True)
        n_orig_full = (ds._full_manifest['aug_index'] == 0).sum()
        ds.resample_augmented(seed=0)
        assert (ds.manifest['aug_index'] == 0).sum() == n_orig_full

    def test_resample_augmented_different_seeds_vary(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_aug_per_window=3)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=True)
        ds.resample_augmented(seed=1)
        idx1 = ds.manifest[ds.manifest['aug_index'] > 0]['aug_index'].tolist()
        ds.resample_augmented(seed=99)
        idx2 = ds.manifest[ds.manifest['aug_index'] > 0]['aug_index'].tolist()
        # With 3 aug copies and 4 windows, different seeds should differ
        assert idx1 != idx2

    def test_resample_augmented_noop_without_column(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        df = pd.read_csv(m)
        df.drop(columns=['aug_index']).to_csv(m, index=False)
        ds = HFLungDataset(manifest_path=m, split='train',
                           include_augmented=False)
        n_before = len(ds)
        ds.resample_augmented(seed=0)
        assert len(ds) == n_before


# ---------------------------------------------------------------------------
# get_train_groups
# ---------------------------------------------------------------------------

class TestGetTrainGroups:

    def test_returns_lists_of_same_length(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_test=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        rec_ids, dates = get_train_groups(manifest_path=m)
        assert isinstance(rec_ids, list)
        assert len(rec_ids) == len(dates)

    def test_only_train_recordings(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_test=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        manifest = pd.read_csv(m)
        test_ids = set(manifest[manifest['split'] == 'test']['recording_id'])
        rec_ids, _ = get_train_groups(manifest_path=m)
        assert not any(r in test_ids for r in rec_ids)

    def test_unique_recording_ids(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6, n_aug_per_window=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        rec_ids, _ = get_train_groups(manifest_path=m)
        assert len(rec_ids) == len(set(rec_ids))

    def test_dates_are_8_digit_strings(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        _, dates = get_train_groups(manifest_path=m)
        assert all(len(d) == 8 and d.isdigit() for d in dates)


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------

class TestGetDataloaders:

    def test_full_mode_returns_train_and_test_only(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=4, n_test=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        loaders = get_dataloaders(manifest_path=m, batch_size=4, num_workers=0)
        assert 'train' in loaders
        assert 'test'  in loaders
        assert 'val'   not in loaders

    def test_fold_mode_includes_val_loader(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6, n_test=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        manifest = pd.read_csv(m)
        ids = manifest[manifest['split'] == 'train']['recording_id'].unique().tolist()
        loaders = get_dataloaders(manifest_path=m, batch_size=4, num_workers=0,
                                  train_recording_ids=ids[:4],
                                  val_recording_ids=ids[4:])
        assert 'val' in loaders

    def test_batch_shape(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=8, n_test=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        loaders = get_dataloaders(manifest_path=m, batch_size=4, num_workers=0)
        imgs, labels = next(iter(loaders['train']))
        assert imgs.shape[1:] == (3, H, W)
        assert labels.shape[1] == 2

    def test_val_contains_only_originals(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=6, n_test=4, n_aug_per_window=2)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        manifest = pd.read_csv(m)
        ids = manifest[manifest['split'] == 'train']['recording_id'].unique().tolist()
        loaders = get_dataloaders(manifest_path=m, batch_size=4, num_workers=0,
                                  train_recording_ids=ids[:4],
                                  val_recording_ids=ids[4:])
        val_manifest = loaders['val'].dataset.manifest
        if 'aug_index' in val_manifest.columns:
            assert (val_manifest['aug_index'] == 0).all()


# ---------------------------------------------------------------------------
# LungSoundClassifier
# ---------------------------------------------------------------------------

class TestLungSoundClassifier:

    def _model(self, freeze_until='layer2'):
        return LungSoundClassifier(variant='resnet18', pretrained=False,
                                   freeze_until=freeze_until, dropout_p=0.0)

    def test_forward_output_shape(self):
        out = self._model()(torch.randn(4, 3, H, W))
        assert out.shape == (4, 2)

    def test_outputs_logits_not_probs(self):
        """Forward pass returns raw logits — verify sigmoid is not applied internally
        by setting head weights large enough to guarantee outputs outside [0, 1]."""
        model = self._model()
        with torch.no_grad():
            model.head.weight.fill_(10.0)
            model.head.bias.fill_(10.0)
        out = model(torch.randn(4, 3, H, W))
        assert (out > 1).any()

    def test_predict_proba_in_zero_one(self):
        probs = self._model().predict_proba(torch.randn(4, 3, H, W))
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_predict_returns_binary(self):
        preds = self._model().predict(torch.randn(4, 3, H, W))
        assert set(preds.unique().tolist()).issubset({0.0, 1.0})

    def test_freeze_layer2_freezes_early_layers(self):
        model = self._model(freeze_until='layer2')
        for p in model.backbone[0].parameters():   # conv1
            assert not p.requires_grad

    def test_freeze_layer2_leaves_layer3_trainable(self):
        model = self._model(freeze_until='layer2')
        assert any(p.requires_grad for p in model.backbone[6].parameters())  # layer3

    def test_freeze_none_all_trainable(self):
        model = LungSoundClassifier(variant='resnet18', pretrained=False,
                                    freeze_until=None)
        assert all(p.requires_grad for p in model.parameters())

    def test_head_always_trainable(self):
        model = self._model(freeze_until='layer4')
        assert all(p.requires_grad for p in model.head.parameters())

    def test_head_output_size(self):
        assert self._model().head.out_features == 2

    def test_resnet18_feature_dim(self):
        assert self._model().head.in_features == 512

    def test_resnet50_feature_dim(self):
        model = LungSoundClassifier(variant='resnet50', pretrained=False)
        assert model.head.in_features == 2048

    def test_trainable_parameters_all_require_grad(self):
        model = self._model()
        assert all(p.requires_grad for p in model.trainable_parameters())

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match='Unknown variant'):
            LungSoundClassifier(variant='resnet999', pretrained=False)

    def test_invalid_freeze_until_raises(self):
        with pytest.raises(ValueError, match='freeze_until must be one of'):
            LungSoundClassifier(variant='resnet18', pretrained=False,
                                freeze_until='layerX')


# ---------------------------------------------------------------------------
# MultiLabelBCELoss
# ---------------------------------------------------------------------------

class TestMultiLabelBCELoss:

    def test_returns_three_scalar_tensors(self):
        loss_fn = MultiLabelBCELoss()
        total, das_l, cas_l = loss_fn(torch.randn(8, 2),
                                      torch.randint(0, 2, (8, 2)).float())
        assert total.shape == das_l.shape == cas_l.shape == ()

    def test_total_is_mean_of_per_label(self):
        loss_fn = MultiLabelBCELoss()
        total, das_l, cas_l = loss_fn(torch.randn(8, 2),
                                      torch.randint(0, 2, (8, 2)).float())
        assert abs(total.item() - (das_l + cas_l).item() / 2) < 1e-5

    def test_all_losses_non_negative(self):
        loss_fn = MultiLabelBCELoss()
        for _ in range(5):
            total, das_l, cas_l = loss_fn(torch.randn(8, 2),
                                          torch.randint(0, 2, (8, 2)).float())
            assert total.item() >= 0
            assert das_l.item() >= 0
            assert cas_l.item() >= 0

    def test_perfect_predictions_give_low_loss(self):
        loss_fn = MultiLabelBCELoss()
        total, _, _ = loss_fn(torch.full((8, 2), 10.0), torch.ones(8, 2))
        assert total.item() < 0.01

    def test_pos_weight_increases_loss_on_positives(self):
        logits = torch.zeros(8, 2)
        labels = torch.ones(8, 2)
        t_no, _, _ = MultiLabelBCELoss()(logits, labels)
        t_w,  _, _ = MultiLabelBCELoss(pos_weight=torch.tensor([5.0, 5.0]))(logits, labels)
        assert t_w.item() > t_no.item()


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:

    def test_round_trip_epoch_and_score(self, tmp_path):
        model = LungSoundClassifier(variant='resnet18', pretrained=False)
        opt   = torch.optim.Adam(model.parameters())
        path  = tmp_path / 'ckpt.pt'
        save_checkpoint(model, opt, epoch=7, score=0.91, path=path)
        assert path.exists()

        model2 = LungSoundClassifier(variant='resnet18', pretrained=False)
        epoch, score = load_checkpoint(path, model2, device=torch.device('cpu'))
        assert epoch == 7
        assert abs(score - 0.91) < 1e-6

    def test_loaded_weights_match_saved(self, tmp_path):
        model = LungSoundClassifier(variant='resnet18', pretrained=False)
        with torch.no_grad():
            model.head.weight.fill_(0.42)
        opt  = torch.optim.Adam(model.parameters())
        path = tmp_path / 'weights.pt'
        save_checkpoint(model, opt, epoch=1, score=0.5, path=path)

        model2 = LungSoundClassifier(variant='resnet18', pretrained=False)
        load_checkpoint(path, model2, device=torch.device('cpu'))

        x = torch.randn(2, 3, H, W)
        model.eval()
        model2.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x), model2(x))

    def test_creates_parent_directories(self, tmp_path):
        model = LungSoundClassifier(variant='resnet18', pretrained=False)
        opt   = torch.optim.Adam(model.parameters())
        path  = tmp_path / 'deep' / 'nested' / 'ckpt.pt'
        save_checkpoint(model, opt, epoch=1, score=0.0, path=path)
        assert path.exists()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_required_keys_present(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'))
        for key in ('mean_auc', 'das_auc', 'cas_auc', 'das_f1', 'cas_f1'):
            assert key in metrics

    def test_auc_values_in_range(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'))
        for key in ('mean_auc', 'das_auc', 'cas_auc'):
            assert 0.0 <= metrics[key] <= 1.0

    def test_f1_values_in_range(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'))
        for key in ('das_f1', 'cas_f1'):
            assert 0.0 <= metrics[key] <= 1.0

    def test_loss_keys_present_when_loss_fn_given(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        loss_fn = MultiLabelBCELoss()
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'), loss_fn=loss_fn)
        for key in ('total_loss', 'das_loss', 'cas_loss'):
            assert key in metrics
            assert metrics[key] > 0

    def test_loss_keys_absent_without_loss_fn(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'), loss_fn=None)
        assert 'total_loss' not in metrics

    def test_mean_auc_is_mean_of_per_label(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False)
        metrics = evaluate(model, tiny_loader(), torch.device('cpu'))
        expected = (metrics['das_auc'] + metrics['cas_auc']) / 2
        assert abs(metrics['mean_auc'] - expected) < 1e-6


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:

    def _setup(self):
        model   = LungSoundClassifier(variant='resnet18', pretrained=False,
                                      freeze_until=None)
        loss_fn = MultiLabelBCELoss()
        opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
        loader  = tiny_loader(n=16, batch_size=8)
        return model, loss_fn, opt, loader

    def test_returns_required_keys(self):
        model, loss_fn, opt, loader = self._setup()
        result = train_one_epoch(model, loss_fn, opt, loader, torch.device('cpu'))
        for key in ('total_loss', 'das_loss', 'cas_loss'):
            assert key in result
            assert result[key] > 0

    def test_weights_updated_after_epoch(self):
        model, loss_fn, opt, loader = self._setup()
        head_before = model.head.weight.detach().clone()
        train_one_epoch(model, loss_fn, opt, loader, torch.device('cpu'))
        assert not torch.allclose(model.head.weight, head_before)

    def test_frozen_weights_unchanged(self):
        """Weights frozen by freeze_until should not change."""
        model = LungSoundClassifier(variant='resnet18', pretrained=False,
                                    freeze_until='layer2')
        frozen_weight = model.backbone[0].weight.detach().clone()  # conv1
        loss_fn = MultiLabelBCELoss()
        opt     = torch.optim.Adam(model.trainable_parameters(), lr=1e-3)
        train_one_epoch(model, loss_fn, opt, tiny_loader(n=8), torch.device('cpu'))
        torch.testing.assert_close(model.backbone[0].weight, frozen_weight)


# ---------------------------------------------------------------------------
# train_run (integration — 2 epochs, tiny synthetic data)
# ---------------------------------------------------------------------------

class TestTrainRun:

    def _patch(self, monkeypatch, tmp_path, m):
        import train as train_mod
        monkeypatch.setattr(train_mod, 'WINDOWS_MANIFEST_PATH', m)
        monkeypatch.setattr(train_mod, 'CHECKPOINTS_DIR', tmp_path / 'ckpts')
        monkeypatch.setattr(train_mod, 'NUM_WORKERS', 0)

    def test_full_mode_returns_keys_and_checkpoint(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=8, n_test=4, n_aug_per_window=1)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        self._patch(monkeypatch, tmp_path, m)

        from train import train_run
        result = train_run(
            run_name='test_full', device=torch.device('cpu'),
            train_recording_ids=None, val_recording_ids=None,
            num_epochs=2, patience=10, batch_size=4,
            learning_rate=1e-3, weight_decay=0.0, pretrained=False,
        )
        assert 'best_score' in result
        assert Path(result['checkpoint']).exists()

    def test_fold_mode_returns_best_epoch(self, tmp_path, monkeypatch):
        # Use 12 train samples so val set (4) has enough class variety for AUC
        m, s = make_manifest(tmp_path, n_train=12, n_test=4, n_aug_per_window=1)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        self._patch(monkeypatch, tmp_path, m)

        manifest = pd.read_csv(m)
        ids = manifest[manifest['split'] == 'train']['recording_id'].unique().tolist()

        from train import train_run
        result = train_run(
            run_name='test_fold', device=torch.device('cpu'),
            train_recording_ids=ids[4:], val_recording_ids=ids[:4],
            num_epochs=2, patience=10, batch_size=4,
            learning_rate=1e-3, weight_decay=0.0, pretrained=False,
        )
        assert 'best_epoch' in result
        assert 'best_score' in result
        assert 0.0 <= result['best_score'] <= 1.0

    def test_history_csv_written(self, tmp_path, monkeypatch):
        m, s = make_manifest(tmp_path, n_train=8, n_test=4)
        monkeypatch.setattr('dataset.SPECTROGRAMS_DIR', s)
        self._patch(monkeypatch, tmp_path, m)

        from train import train_run
        train_run(
            run_name='hist_test', device=torch.device('cpu'),
            num_epochs=2, patience=10, batch_size=4,
            learning_rate=1e-3, weight_decay=0.0, pretrained=False,
        )
        history_path = tmp_path / 'ckpts' / 'hist_test' / 'history.csv'
        assert history_path.exists()
        df = pd.read_csv(history_path)
        assert len(df) == 2
        assert 'epoch' in df.columns and 'train_loss' in df.columns
