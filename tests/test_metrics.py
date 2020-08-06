# Provided under the CC-BY-SA license. Please cite the accompanying paper when using TOP dataset -
# @ARTICLE {
#     author  = "Sonal Gupta and Rushin Shah and Mrinal Mohit and Anuj Kumar and Michael Lewis",
#     title   = "Semantic Parsing for Task Oriented Dialog using Hierarchical Representations",
#     journal = "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
#     year    = "2018",
#     month   = "Oct"
# }
import unittest

from new_semantic_parsing import metrics
from new_semantic_parsing.schema_tokenizer import TopSchemaTokenizer


class TestMetrics(unittest.TestCase):
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

        _expected_paths = {
            "IN:GET_ESTIMATED_DEPARTURE.SL:SOURCE": "[IN:GET_LOCATION_HOME [SL:CONTACT my] house]",
            "IN:GET_ESTIMATED_DEPARTURE.SL:SOURCE.IN:GET_LOCATION_HOME.SL:CONTACT": "my",
            "IN:GET_ESTIMATED_DEPARTURE.SL:DESTINATION": "[IN:GET_LOCATION [SL:POINT_ON_MAP the Hamilton Mall]]",
            "IN:GET_ESTIMATED_DEPARTURE.SL:DESTINATION.IN:GET_LOCATION.SL:POINT_ON_MAP": "the Hamilton Mall",
            "IN:GET_ESTIMATED_DEPARTURE.SL:DATE_TIME_ARRIVAL": "right when it opens on Saturday",
        }

        # fmt: on

        tree = metrics.Tree.from_tokens(TopSchemaTokenizer.tokenize(_input))
        _paths = metrics._get_paths_with_values(tree)

        self.assertEqual(_paths, _expected_paths)

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
            "tree_path_precision": 3/6.,
            "tree_path_recall": 3/5.,
            "tree_path_f1": 0.5455,
        }

        pred4 = (
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
                "[SL:DATE_TIME_ARRIVAL wrong text ] "
            "]")

        expected4 = {
            "tree_path_precision": 4/5.,
            "tree_path_recall": 4/5.,
            "tree_path_f1": 4/5,
        }

        # fmt: on

        true_tokens = TopSchemaTokenizer.tokenize(true)

        for i, (pred, expected) in enumerate(
            zip([pred1, pred2, pred3, pred4], [expected1, expected2, expected3, expected4])
        ):
            with self.subTest(i):
                pred_tokens = TopSchemaTokenizer.tokenize(pred)
                res = metrics.get_tree_path_scores([pred_tokens], [true_tokens])
                res = {k: round(v, 4) for k, v in res.items()}
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

    def test_build_tree(self):
        # example from TOP dataset arxiv.org/abs/1810.07942
        # fmt: off

        schema = (
            "[IN:GET_DISTANCE How far is "
                "[SL:DESTINATION "
                    "[IN:GET_RESTAURANT_LOCATION the "
                        "[SL:FOOD_TYPE coffee] "
                        "shop "
                    "]"
                "]"
            "]")

        expected_tree = {
            'IN:GET_DISTANCE': [
                {'SL:DESTINATION': [
                    {'IN:GET_RESTAURANT_LOCATION': [
                        {'the': []},
                        {'SL:FOOD_TYPE': [{'coffee': []}]},
                        {'shop': []}
                    ]}
                ]}
            ]
        }

        # fmt: on

        tree = metrics.Tree.from_tokens(TopSchemaTokenizer.tokenize(schema))

        self.assertDictEqual(expected_tree, tree._dict_repr)
