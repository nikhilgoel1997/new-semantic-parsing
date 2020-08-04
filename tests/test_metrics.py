# Provided under the CC-BY-SA license. Please cite the accompanying paper when using TOP dataset -
# @ARTICLE {
#     author  = "Sonal Gupta and Rushin Shah and Mrinal Mohit and Anuj Kumar and Michael Lewis",
#     title   = "Semantic Parsing for Task Oriented Dialog using Hierarchical Representations",
#     journal = "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
#     year    = "2018",
#     month   = "Oct"
# }
import unittest
from collections import Counter

from new_semantic_parsing import metrics
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer


class TestMetrics(unittest.TestCase):
    def test_get_slot_paths(self):
        test_case_1 = {
            "input": "[ IN:INTENT [ SL:SLOT1 [ SL:SLOT2 slot value ] ] ]",
            "output": Counter(["SLOT1.SLOT2"]),
        }

        test_case_2 = {
            "input": (
                "[ IN:INTENT [ SL:SLOT1 [ SL:SLOT2 slot value ] ] [ SL:SLOT1 [ SL:SLOT2 slot"
                " value]]]"
            ),
            "output": Counter({"SLOT1.SLOT2": 2}),
        }

        test_case_3 = {
            "input": (
                "[ IN:INTENT "
                "[ IN:GET_SLOT"
                "[ SL:SLOT1 "
                "[ SL:SLOT2 slot value ] "
                "[ SL:SLOT1 [ SL:SLOT2 slot value ] ] "
                "[ SL:SLOT3 slot3value ]"
                "] "
                "]"
                "]"
            ),
            "output": Counter({"SLOT1.SLOT2": 1, "SLOT1.SLOT1.SLOT2": 1, "SLOT1.SLOT3": 1}),
        }

        for i, test_case in enumerate([test_case_1, test_case_2, test_case_3]):
            with self.subTest(i):
                _tokens = TopSchemaTokenizer.tokenize(test_case["input"])
                slot_paths = metrics._get_slot_paths(metrics.Tree.from_tokens(_tokens))
                self.assertEqual(
                    slot_paths, test_case["output"], msg=(test_case["input"], slot_paths)
                )

    def test_get_paths_with_values(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        # fmt: off
        _input = (
            "[IN:GET_ESTIMATED_DEPARTURE When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:GET_LOCATION "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")

        _output = {
            "IN:GET_ESTIMATED_DEPARTURE.SL:SOURCE.IN:GET_LOCATION_HOME.SL:CONTACT": "my",
            "IN:GET_ESTIMATED_DEPARTURE.SL:DESTINATION.IN:GET_LOCATION.SL:POINT_ON_MAP": "the Hamilton Mall",
            "IN:GET_ESTIMATED_DEPARTURE.SL:DATE_TIME_ARRIVAL": "right when it opens on Saturday",
        }
        # fmt: on

        tree = metrics.Tree.from_tokens(TopSchemaTokenizer.tokenize(_input))
        self.assertEqual(metrics._get_paths_with_values(tree), _output)

    def test_get_tree_path_scores(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        # fmt: off

        true = (
            "[IN:GET_ESTIMATED_DEPARTURE When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:GET_LOCATION "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")

        pred1 = true

        expected1 = {
            "tree_path_precision": 1.0,
            "tree_path_recall": 1.0,
            "tree_path_f1": 1.0,
        }

        pred2 = (
            "[IN:WRONG_INTENT When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:GET_LOCATION "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")

        expected2 = {
            "tree_path_precision": 0.0,
            "tree_path_recall": 0.0,
            "tree_path_f1": 0.0,
        }

        pred3 = (
            "[IN:GET_ESTIMATED_DEPARTURE When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:WRONG_SLOT "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:WRONG_SLOT right when it opens on Saturday ] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")

        expected3 = {
            "tree_path_precision": 1/2.,
            "tree_path_recall": 2/3.,
            "tree_path_f1": 0.5714285714285715,
        }

        # fmt: on

        true_tokens = TopSchemaTokenizer.tokenize(true)

        for i, (pred, expected) in enumerate(
            zip([pred1, pred2, pred3], [expected1, expected2, expected3])
        ):
            with self.subTest(i):
                pred_tokens = TopSchemaTokenizer.tokenize(pred)
                res = metrics.get_tree_path_scores([pred_tokens], [true_tokens])
                self.assertDictEqual(expected, res)

    def test_get_tree_path_scores_for_classes(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        # fmt: off

        true = (
            "[IN:GET_ESTIMATED_DEPARTURE When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:GET_LOCATION "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")

        pred = (
            "[IN:GET_ESTIMATED_DEPARTURE When should I leave "
                "[SL:SOURCE "
                    "[IN:GET_LOCATION_HOME "
                        "[SL:CONTACT my ] house "
                    "] "
                "] to get to "
                "[SL:DESTINATION "
                    "[IN:WRONG_SLOT "
                        "[SL:POINT_ON_MAP the Hamilton Mall ] "
                    "] "
                "] "
                "[SL:WRONG_SLOT right when it opens on Saturday ] "
                "[SL:DATE_TIME_ARRIVAL right when it opens on Saturday ] "
            "]")
        # fmt: on

        classes1 = ["IN:GET_LOCATION_HOME"]

        expected1 = {
            "tree_path_precision": 1.0,
            "tree_path_recall": 1.0,
            "tree_path_f1": 1.0,
        }

        true_tokens = TopSchemaTokenizer.tokenize(true)

        for i, (classes, expected) in enumerate(zip([classes1], [expected1])):
            with self.subTest(i):
                pred_tokens = TopSchemaTokenizer.tokenize(pred)
                res = metrics.get_tree_path_scores([pred_tokens], [true_tokens], classes)
                self.assertDictEqual(expected, res)
