# Provided under the CC-BY-SA license. Please cite the accompanying paper when using TOP dataset -
# @ARTICLE {
#     author  = "Sonal Gupta and Rushin Shah and Mrinal Mohit and Anuj Kumar and Michael Lewis",
#     title   = "Semantic Parsing for Task Oriented Dialog using Hierarchical Representations",
#     journal = "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
#     year    = "2018",
#     month   = "Oct"
# }
import unittest

import torch

from new_semantic_parsing import utils
from new_semantic_parsing.metrics import compute_metrics_from_batch


class TopSchemaGetVocabularyTest(unittest.TestCase):
    def test_get_vocab(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        schema_str = (
            "[IN:GET_ESTIMATED_ARRIVAL What time will I arrive at "
            "[SL:DESTINATION [IN:GET_LOCATION_HOME [SL:CONTACT_RELATED "
            "my ] [SL:TYPE_RELATION Mom ] 's house ] ] if I leave "
            "[SL:DATE_TIME_DEPARTURE in five minutes ] ? ]"
        )
        schema_voc = {
            "[",
            "]",
            "IN:",
            "SL:",
            "CONTACT_RELATED",
            "DATE_TIME_DEPARTURE",
            "DESTINATION",
            "GET_ESTIMATED_ARRIVAL",
            "GET_LOCATION_HOME",
            "TYPE_RELATION",
        }

        res = utils.get_vocab_top_schema(schema_str)
        self.assertSetEqual(res, schema_voc)


class TestGetModelType(unittest.TestCase):
    def test_model_type(self):
        model_type = utils.get_model_type("distilbert-base-uncased")
        self.assertEqual(model_type, "distilbert")


class TestGetMetircsTorch(unittest.TestCase):
    def test_metrics_computation(self):
        x = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 0, 0, 0],
                [1, 3, 5, 7, 9, 0, 0, 0, 0],
                [19, 18, 17, 16, 18, 13, 19, 0, 0],
            ]
        )

        y = torch.tensor(
            [
                [3, 2, 8, 4, 5, 5, 0, 0, 0],
                [1, 3, 5, 7, 9, 0, 0, 0, 0],
                [19, 8, 17, 16, 18, 5, 1, 0, 0],
            ]
        )

        m = torch.tensor(
            [
                [0, 1, 0, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0],
            ]
        )

        assert x.shape == y.shape
        assert y.shape == m.shape

        stop_tokens = [5, 0]
        metrics = compute_metrics_from_batch(x, y, m, stop_tokens)

        # NOTE: we expect micro averaging
        expected_accuracy = 0.91666666666
        expected_exact_match = 0.33333333333  # mask is not used for EM, only truncation
        expected_intent_precision = 0.666666666

        self.assertAlmostEqual(metrics["accuracy"].numpy(), expected_accuracy)
        self.assertAlmostEqual(metrics["exact_match"].numpy(), expected_exact_match)
        self.assertAlmostEqual(
            metrics["first_intent_precision"].numpy(), expected_intent_precision
        )


class TestSnips(unittest.TestCase):
    def test_snips2top(self):
        ex1 = [
            {"text": "Weather "},
            {"text": "tomorrow", "entity": "timeRange"},
            {"text": " in "},
            {"text": "Lowell", "entity": "geographic_poi"},
        ]
        intent1 = "GetWeather"
        expected_text = "Weather tomorrow in Lowell"
        expected_schema = (
            "[IN:GETWEATHER Weather [SL:TIMERANGE tomorrow ] in [SL:GEOGRAPHIC_POI Lowell ] ]"
        )

        out_text, out_schema = utils.snips2top(ex1, intent1)

        self.assertSequenceEqual(expected_text, out_text)
        self.assertSequenceEqual(expected_schema, out_schema)
