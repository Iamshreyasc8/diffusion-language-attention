"""Optional real-library integration, with tiny RANDOM weights and no downloads.

Run ATTENTION_MODEL_SMOKE=1 python -m unittest discover -s tests -v in the
model environment. This checks plumbing, not pretrained model quality.
"""
import json
import os
from pathlib import Path
import string
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ATTENTION_MODEL_SMOKE') == '1', 'optional model-stack integration')
class ModelSmokeTest(unittest.TestCase):
    def test_trace_export_and_metadata(self):
        import torch
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusion_attention.demo import run_example
        torch.set_num_threads(2)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = {'<|startoftext|>': 0, '<|endoftext|>': 1}
            for character in string.ascii_lowercase:
                for token in (character, character + '</w>'):
                    vocab[token] = len(vocab)
            (root / 'vocab.json').write_text(json.dumps(vocab))
            (root / 'merges.txt').write_text('#version: 0.2\n')
            tokenizer = CLIPTokenizer(str(root / 'vocab.json'), str(root / 'merges.txt'), model_max_length=77)
            text_encoder = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocab), hidden_size=32,
                intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
            unet = UNet2DConditionModel(sample_size=64, in_channels=4, out_channels=4,
                down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
                up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(16, 32),
                layers_per_block=1, cross_attention_dim=32, attention_head_dim=4, norm_num_groups=8)
            vae = AutoencoderKL(in_channels=3, out_channels=3,
                down_block_types=('DownEncoderBlock2D',) * 4,
                up_block_types=('UpDecoderBlock2D',) * 4,
                block_out_channels=(8, 8, 16, 16), layers_per_block=1,
                latent_channels=4, norm_num_groups=8, sample_size=512)
            # No pretrained model or safety component in this random-weight test fixture.
            # The actual loader retains the pretrained pipeline's safety checker.
            pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                unet=unet, scheduler=DDIMScheduler(num_train_timesteps=10),
                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
            pipe.set_progress_bar_config(disable=True)
            output = root / 'result'
            report = run_example(pipe, 'a red car', 'red car', output,
                                 steps=1, model_id='random-weight-test-fixture', device='cpu')
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['pairwise_scores']), 1)
            self.assertTrue(0 <= report['max_iou'] <= 1)
            for name in ['generated.png', 'attention.png', 'heatmaps.npz', 'scores.json']:
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(json.loads((output / 'scores.json').read_text())['seed'], 42)
            with self.assertRaises(FileExistsError):
                run_example(pipe, 'a red car', 'red car', output, steps=1)


if __name__ == '__main__':
    unittest.main()
