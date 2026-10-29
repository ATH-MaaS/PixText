"""Model layouts used by portable installs and extra_model_paths.yaml."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from comfyui.model_paths import discover_models, discover_packages


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / "ComfyUI_windows_portable" / "ComfyUI" / "models"
        self.registered = self.models / "pixtext"

    @staticmethod
    def package(root, name):
        path = root / name
        path.mkdir(parents=True)
        (path / "manifest.json").write_text("{}")
        (path / "model.safetensors").touch()
        return path

    def discover(self, roots=None):
        return discover_packages(roots if roots is not None else [self.registered], self.models)

    def test_portable_models_directly_under_models_are_found(self):
        expected = {
            name: self.package(self.models, name)
            for name in ("PixText-Teacher", "PixText-MeanFlow")
        }
        self.assertEqual(self.discover(), expected)

    def test_existing_pixtext_layout_is_preserved(self):
        package = self.package(self.registered, "PixText-MeanFlow")
        self.assertEqual(self.discover(), {"PixText-MeanFlow": package})

    def test_extra_model_paths_precede_models_root(self):
        extra = self.root / "shared-models"
        package = self.package(extra, "PixText-Teacher")
        self.package(self.models, "PixText-Teacher")
        self.assertEqual(self.discover([self.registered, extra]), {"PixText-Teacher": package})

    def test_existing_package_wins_duplicate_names(self):
        package = self.package(self.registered, "PixText-MeanFlow")
        self.package(self.models, "PixText-MeanFlow")
        self.assertEqual(self.discover(), {"PixText-MeanFlow": package})

    def test_incomplete_downloads_are_not_listed(self):
        for root, name in ((self.models, "PixText-Teacher"), (self.registered, "partial")):
            path = self.package(root, name)
            (path / "model.safetensors").unlink()
        weights_only = self.models / "PixText-MeanFlow"
        weights_only.mkdir()
        (weights_only / "model.safetensors").touch()
        self.assertEqual(self.discover(), {})

    def test_unrelated_subtrees_are_not_recursively_searched(self):
        self.package(self.models / "other-node" / "cache", "PixText-Teacher")
        self.assertEqual(self.discover(), {})

    def test_new_download_appears_without_restarting_discovery(self):
        self.assertEqual(self.discover(), {})
        package = self.package(self.models, "PixText-Teacher")
        self.assertEqual(self.discover(), {"PixText-Teacher": package})

    def test_standalone_weights_have_native_names_and_legacy_api_aliases(self):
        self.registered.mkdir(parents=True)
        expected = {}
        for name in ("PixText-Teacher", "PixText-MeanFlow"):
            path = self.registered / (name + ".safetensors")
            path.touch()
            expected[path.name] = path
            expected[name] = path
        self.assertEqual(discover_models([self.registered], self.models), expected)

    def test_complete_packages_also_satisfy_native_workflow_names(self):
        path = self.package(self.models, "PixText-MeanFlow")
        choices = discover_models([self.registered], self.models)
        self.assertEqual(choices["PixText-MeanFlow.safetensors"], path)
        self.assertEqual(choices["PixText-MeanFlow"], path)

    def test_registered_standalone_weights_precede_fallback_packages(self):
        self.package(self.models, "PixText-Teacher")
        self.registered.mkdir()
        path = self.registered / "PixText-Teacher.safetensors"
        path.touch()
        choices = discover_models([self.registered], self.models)
        self.assertEqual(choices["PixText-Teacher.safetensors"], path)

    def test_unrecognized_standalone_weights_are_not_offered(self):
        self.models.mkdir(parents=True)
        (self.models / "some-other-model.safetensors").touch()
        self.assertEqual(discover_models([self.registered], self.models), {})
