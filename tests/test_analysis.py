import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from diffusion_attention.analysis import (
    as_heatmap, binary_mask, collect_heatmaps, continuous_iou,
    expression_words, pairwise_scores, validate_prompt,
)
from diffusion_attention.demo import check_token_budget, read_examples, select_device, save_visuals


class TensorLike:
    def __init__(self, array): self.array = array
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


class AnalysisTests(unittest.TestCase):
    def test_array_and_tensor_inputs(self):
        expected = np.array([[1., 2.]])
        class WordMap:
            heatmap = TensorLike(expected)
        np.testing.assert_equal(as_heatmap(expected), as_heatmap(WordMap()))

    def test_iou_known_values(self):
        self.assertEqual(continuous_iou([[1, 0]], [[0, 1]]), 0)
        self.assertEqual(continuous_iou([[1, 2]], [[1, 2]]), 1)
        self.assertAlmostEqual(continuous_iou([[1, 1]], [[1, 0]]), 0.5)
        self.assertEqual(continuous_iou([[0, 0]], [[0, 0]]), 0)

    def test_iou_is_symmetric_and_common_scale_invariant(self):
        a, b = np.array([[1., 3.]]), np.array([[2., 4.]])
        self.assertAlmostEqual(continuous_iou(a, b), continuous_iou(b, a))
        self.assertAlmostEqual(continuous_iou(a, b), continuous_iou(a * 1e300, b * 1e300))

    def test_invalid_heatmaps(self):
        for value in ([], [1, 2], [[np.nan]], [[np.inf]], [[-1]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                as_heatmap(value)
        with self.assertRaises(ValueError):
            continuous_iou([[1]], [[1, 2]])

    def test_constant_maps_produce_empty_masks(self):
        for value in (np.zeros((2, 2)), np.ones((2, 2))):
            self.assertFalse(binary_mask(value).any())

    def test_sparse_mask_excludes_zero_background(self):
        value = np.zeros((10, 10)); value[0, 0] = 1
        self.assertEqual(binary_mask(value).sum(), 1)
        self.assertEqual(binary_mask(value, 0).sum(), 1)

    def test_percentile_validation(self):
        for value in (-1, 101, np.nan):
            with self.assertRaises(ValueError): binary_mask([[0, 1]], value)

    def test_words_deduplicated_with_order(self):
        self.assertEqual(expression_words('Fire truck FIRE'), ['fire', 'truck'])
        for value in ('', '!!!', None):
            with self.assertRaises(ValueError): expression_words(value)

    def test_prompt_validation(self):
        self.assertEqual(validate_prompt('A FIRE truck.', 'fire truck'), ['fire', 'truck'])
        with self.assertRaises(ValueError): validate_prompt('A truck.', 'fire truck')
        with self.assertRaises(ValueError): validate_prompt('A firetruck.', 'truck')
        with self.assertRaises(ValueError): validate_prompt('', 'truck')

    def test_missing_words_are_not_zero_heatmaps(self):
        class Global:
            def compute_word_heat_map(self, word):
                if word == 'missing': raise ValueError('missing token')
                return np.ones((2, 2))
        maps, missing = collect_heatmaps(Global(), ['truck', 'missing'])
        self.assertEqual(list(maps), ['truck'])
        self.assertEqual(list(missing), ['missing'])
        self.assertEqual(pairwise_scores(maps), [])

    def test_pairwise_scores(self):
        maps = {word: np.ones((2, 2)) for word in ['a', 'b', 'c']}
        scores = pairwise_scores(maps)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(row['continuous_iou'] == 1 for row in scores))
        json.dumps(scores, allow_nan=False)

    def test_device_fallback(self):
        class Torch:
            class cuda:
                @staticmethod
                def is_available(): return False
        self.assertEqual(select_device('auto', Torch), 'cpu')
        with self.assertRaises(RuntimeError): select_device('cuda', Torch)

    def test_overlong_prompt_rejected(self):
        class Tokenizer:
            model_max_length = 3
            def __call__(self, text, **kwargs): return {'input_ids': list(text)}
        self.assertEqual(check_token_budget(Tokenizer(), 'abc'), 3)
        with self.assertRaises(ValueError): check_token_budget(Tokenizer(), 'abcd')

    def test_sample_data_and_invalid_rows(self):
        rows = read_examples(Path(__file__).resolve().parents[1] / 'examples/prompts.csv')
        self.assertEqual(len(rows), 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            for content in ('prompt,expression\n', 'prompt,expression\ntruck,fire truck\n', 'wrong,columns\nx,y\n'):
                path.write_text(content)
                with self.assertRaises(ValueError): read_examples(path)

    def test_visual_export_handles_missing_and_rectangular_maps(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = Image.new('RGB', (80, 40), 'white')
            save_visuals(image, {}, directory, 90)
            save_visuals(image, {'truck': np.array([[0., 1., 0.], [0., 0., 0.]])}, directory, 90)
            with Image.open(directory / 'generated.png') as saved:
                self.assertEqual(saved.size, (80, 40))
            self.assertTrue((directory / 'attention.png').is_file())


if __name__ == '__main__':
    unittest.main()
