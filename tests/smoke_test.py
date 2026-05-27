#!/usr/bin/env python3
import os
import sys
import torch
import unittest
from unittest.mock import MagicMock, patch

# Ensure src/ is importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config.hardware import is_cuda_available, is_bf16_supported
from src.config.training import get_training_args
from src.evaluation.vector_engine import EscoVectorEvaluator

class PipelineSmokeTests(unittest.TestCase):
    
    def test_hardware_precision_logic(self):
        """Verify that training args correctly toggle bf16/fp16 based on hardware."""
        args = get_training_args()
        cuda_av = is_cuda_available()
        has_bf16 = is_bf16_supported()
        
        if cuda_av:
            if has_bf16:
                self.assertTrue(args.bf16)
                self.assertFalse(args.fp16)
            else:
                self.assertFalse(args.bf16)
                self.assertTrue(args.fp16)
            self.assertEqual(args.optim, "adamw_8bit")
        else:
            self.assertFalse(args.bf16)
            self.assertFalse(args.fp16)
            self.assertEqual(args.optim, "adamw_torch")
            
    @patch("src.evaluation.vector_engine.extract_unique_targets")
    def test_vector_evaluator_matrix(self, mock_extract):
        """Mock unit test for EscoVectorEvaluator tensor shape broadcast and inference."""
        # Mock target DB
        mock_extract.return_value = (["Data Scientist", "Software Engineer"], ["2511", "2512"])
        
        # Mock tokenizer inputs
        mock_inputs = MagicMock()
        mock_inputs.attention_mask = torch.ones((1, 5), dtype=torch.long)
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "Mocked prompt"
        mock_tokenizer.return_value = MagicMock(
            to=MagicMock(return_value=mock_inputs)
        )
        
        # Mock eval records
        eval_records = [{"input": "python, sql, ml", "output": "ESCO Occupation Title: data scientist\nISCO-08 Code: 2511"}]
        evaluator = EscoVectorEvaluator(eval_records, tokenizer=mock_tokenizer)
        
        # Mock Model output
        mock_model = MagicMock()
        mock_output = MagicMock()
        
        # Simulating hidden states: (batch_size, sequence_length, hidden_dim)
        hidden_states = [torch.rand(1, 5, 128)] 
        mock_output.hidden_states = hidden_states
        mock_model.return_value = mock_output
        
        # Force the evaluator to run in simulated CUDA mode to trigger the vector math block
        with patch("src.evaluation.vector_engine.is_cuda_available", return_value=True):
            # 1. Test embedding precomputation matrix creation
            evaluator._compute_target_embeddings(mock_model)
            
            # Should cache 2 embeddings (since we mocked 2 target titles)
            self.assertIsNotNone(evaluator.target_embeddings)
            self.assertEqual(evaluator.target_embeddings.shape, (2, 128))
            self.assertTrue(evaluator.target_embeddings.is_contiguous(), "Target matrix is not strictly contiguous!")
            
            # 2. Test execution of the broadcasting loop
            precision, recall, f1 = evaluator.run_evaluation(mock_model)
            
            # Validate output bounds
            self.assertGreaterEqual(precision, 0.0)
            self.assertGreaterEqual(recall, 0.0)
            self.assertGreaterEqual(f1, 0.0)

if __name__ == "__main__":
    unittest.main()
